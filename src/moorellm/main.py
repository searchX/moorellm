from functools import wraps
from typing import Callable, Dict, Any, Literal, Optional, Type, Union
from pydantic import BaseModel, ValidationError, create_model
import openai
import jinja2

# Setup logging
import logging

logger = logging.getLogger("moorellm")

from moorellm.utils import _add_transitions, _create_response_model
from moorellm.models import MooreRun, MooreState, DefaultResponse, StateMachineError


class MooreFSM:
    """Moore Finite State Machine based LLM Agent class, this allows to define states and transitions and use latest
    structured response from OpenAI/Azure OpenAI API

    :param initial_state: Initial state of the FSM
    :param end_state: End state of the FSM, default is "END" (currently not used)
    :type initial_state: str
    :type end_state: str
    :return: MooreFSM object
    :rtype: :class:`moorellm.main.MooreFSM`
    """

    def __init__(self, initial_state: str, end_state: str = "END"):
        """Initialize the Moore FSM with initial state and end state"""
        self._state = initial_state
        self._next_state = None
        self._initial_state = initial_state
        self._end_state = end_state
        self._states = {}
        self._chat_history = []
        self._full_chat_history = []
        self.user_defined_context = {}

    def state(
        self,
        state_key: str,
        system_prompt: str,
        temperature: float = 0.5,
        transitions: Dict[str, str] = {},
        response_model: Optional[BaseModel] = None,
        pre_process_input: Optional[Callable] = None,
        pre_process_chat: Optional[Callable] = None,
        pre_process_system_prompt: Optional[Callable] = None,
    ):
        """Decorator to add a state to the FSM while defining transitions and other details.

        :param state_key: Key of this state
        :param system_prompt: System prompt for this state
        :param temperature: Temperature for the completion
        :param transitions: Transitions to other states, defined by key-value pairs of next_state_key: condition
        :param response_model: Pydantic model for response parsing, default is None (ie give string response)
        :param pre_process_input: Pre-process user input before sending to OpenAI API
        :param pre_process_chat: Pre-process chat history before sending to OpenAI API
        :param pre_process_system_prompt: Pre-process system prompt before sending to OpenAI API
        :type state_key: str
        :type system_prompt: str
        :type temperature: float
        :type transitions: Dict[str, str]
        :type response_model: BaseModel, optional
        :type pre_process_input: Callable, optional
        :type pre_process_chat: Callable, optional
        :type pre_process_system_prompt: Callable, optional
        :return: Decorator function
        :rtype: Callable

        .. code-block:: python

            @fsm.state(
                state_key="START",
                system_prompt="Hello, how can I help you?",
                temperature=0.5,
                transitions={"END": "when user has said quit or exit"},
                response_model=DefaultResponse,
            )
            async def start_state(fsm: MooreFSM, response: DefaultResponse, will_transition: bool):
                pass

        .. note:: The state function should be an async function and should have the following signature:

            .. code-block:: python

                    async def state_function(fsm: MooreFSM, response: Any, will_transition: bool):
                        pass

        .. note:: The response model should be a Pydantic model, if not defined then the response will be a string.
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            # Register the state function
            self._states[state_key] = MooreState(
                key=state_key,
                func=wrapper,
                system_prompt=system_prompt,
                temperature=temperature,
                transitions=transitions,
                response_model=response_model,
                pre_process_input=pre_process_input,
                pre_process_chat=pre_process_chat,
                pre_process_system_prompt=pre_process_system_prompt,
            )
            return wrapper

        return decorator

    async def run(
        self,
        async_openai_instance: Union[openai.AsyncAzureOpenAI, openai.AsyncOpenAI],
        user_input: str,
        model: str = "gpt-4o-2024-08-06",
        *args,
        **kwargs,
    ) -> MooreRun:
        """Run the FSM with user input and get the response from OpenAI API, only one iteration is done at a time.

        :param async_openai_instance: OpenAI/AzureOpenAI instance to use for completion
        :param user_input: User input to the FSM
        :param model: Model to use for completion, default is "gpt-4o-2024-08-06"
        :type async_openai_instance: Union[openai.AsyncAzureOpenAI, openai.AsyncOpenAI]
        :type user_input: str
        :type model: str
        :return: MooreRun object
        :rtype: :class:`moorellm.models.MooreRun`

        .. code-block:: python

            while True:
                user_input = input("You: ")
                run: MooreRun = await fsm.run(async_openai_instance, user_input)
                print(f"AI: {run.response}")
                if run.state == "END":
                    break

        .. note:: The run function should be called in a loop to keep the FSM running, it will return the details from the run.

        .. note:: Only use AsyncOpenAI or AsyncAzureOpenAI instance for async completion.
        """
        logger.debug(f"Current state: {self._state}, User input: {user_input}")

        # Get the current state
        current_state: MooreState = self._states.get(self._state, None)
        if not current_state:
            logger.error(StateMachineError(f"State {self._state} not found in states."))

        if current_state.pre_process_input:
            user_input = current_state.pre_process_input(user_input, self) or user_input
            logger.debug(f"Pre-processed user input: {user_input}")

        state_system_prompt = current_state.system_prompt

        # Pre-process system prompt with Jinja2
        template = jinja2.Template(state_system_prompt)
        state_system_prompt = template.render(self.user_defined_context)

        if current_state.pre_process_system_prompt:
            state_system_prompt = (
                current_state.pre_process_system_prompt(state_system_prompt, self)
                or state_system_prompt
            )

        processed_system_prompt = _add_transitions(state_system_prompt, current_state)
        logger.debug(f"Processed system prompt: {processed_system_prompt}")

        # Add user input to chat history
        self._chat_history.append({"role": "user", "content": user_input})
        self._full_chat_history.append({"role": "user", "content": user_input})

        # First create a good chat history
        system_prompt_lined = {"role": "system", "content": processed_system_prompt}
        chat_history_copy = [system_prompt_lined] + self._chat_history.copy()

        # Pre-process chat if needed
        if current_state.pre_process_chat:
            chat_history_copy = current_state.pre_process_chat(chat_history_copy, self)

        # Now let's try to call openai function
        logger.debug(f"Getting completion for model: {model}")

        output_response_model = _create_response_model(
            current_state.response_model, current_state.transitions, current_state.key
        )

        completion = await async_openai_instance.beta.chat.completions.parse(
            model=model,
            messages=chat_history_copy,
            response_format=output_response_model,
        )

        message = completion.choices[0].message
        if not message.parsed:
            raise StateMachineError(
                f"Error in parsing the completion: {message.refusal}"
            )

        # Extract the response and next state key
        response_dict = message.parsed.model_dump()

        # Default to current state if no next state key
        next_state_key = response_dict.get("next_state_key", current_state.key)

        # Also get the response
        response = response_dict.get("response", None)

        # If no model was specificed then get content from response
        if not current_state.response_model:
            response = response.get("content", response)
        else:
            # Parse the response model into pydantic model
            try:
                response = current_state.response_model(**response)
            except ValidationError as e:
                raise StateMachineError(f"Error in parsing response model: {e}")

        # Check if next state key is valid
        if next_state_key not in self._states:
            logger.error(StateMachineError(f"Next state {next_state_key} not found in states, resetting to current state."))
            next_state_key = current_state.key

        self._next_state = next_state_key
        cached_next_state = next_state_key

        # Now we call function with all the details, but before that create function context params
        function_context = {
            "fsm": self,
            "response": response,
            "will_transition": self._state != self._next_state,
            **kwargs,
        }

        # Call the function
        final_response = await current_state.func(**function_context)
        final_response_str = ""
        if final_response:
            # Add the response to chat history
            final_response_str = final_response
        else:
            if not current_state.response_model:
                final_response_str = response
            else:
                final_response_str = response
                raise StateMachineError(
                    "Response model is defined but no response returned from the state function."
                )

        logger.debug(f"Final response: {final_response_str}")

        # Add the response to chat history
        self._chat_history.append({"role": "assistant", "content": final_response_str})
        self._full_chat_history.append({"role": "assistant", "content": final_response_str})

        # Update the state
        self._state = self._next_state

        if self._next_state != cached_next_state:
            logger.debug(f"Manually transitioned to next state: {self._state}")
        else:
            # TODO: Use guard rails to check if the transition is valid
            logger.debug(f"Transitioned to next state: {self._state}")

        self._next_state = None

        # Return the run object
        return MooreRun(
            state=self._state,
            chat_history=chat_history_copy,
            context_data=self.user_defined_context,
            response_raw=response_dict,
            response=final_response_str,
        )

    def reset(self):
        """Reset the FSM to initial state."""
        self._state = self._initial_state
        self._next_state = None
        self._chat_history = []
        self.user_defined_context = {}
        logger.debug("FSM reset to initial state.")

    def get_next_state(self):
        """Get the next state."""
        return self._next_state

    def get_current_state(self):
        """Get the current state."""
        return self._state

    def set_next_state(self, next_state: str):
        """Set the next state."""
        self._next_state = next_state
        logger.debug(f"Manually set next state: {self._next_state}")

    def get_chat_history(self):
        """Get the chat history."""
        return self._chat_history
    
    def set_chat_history(self, chat_history: list):
        """Set the chat history."""
        self._chat_history = chat_history
        logger.debug(f"Chat history set: {chat_history}")

    def get_full_chat_history(self):
        """Get the full chat history."""
        return self._full_chat_history

    def set_context_data(self, key: str, value: Any):
        """Set data into user defined context."""
        self.user_defined_context[key] = value
        logger.debug(f"User defined context set: {key}={value}")

    def set_context_data_dict(self, data: Dict[str, Any]):
        """Set data into user defined context."""
        self.user_defined_context.update(data)
        logger.debug(f"User defined context set: {data}")

    def get_context_data(self, key: str, default: Any = None):
        """Get data from user defined context."""
        return self.user_defined_context.get(key, default)

    def get_full_context_data(self):
        """Get full user defined context."""
        return self.user_defined_context

    def is_completed(self):
        """Check if the FSM is completed."""
        return self._state == self._end_state
