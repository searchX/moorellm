from typing import Any, Callable, Optional, Type
import pydantic
from pydantic import BaseModel


class MooreState(pydantic.BaseModel):
    """
    Represents a state in the Finite State Machine (FSM) for managing the AI's conversation flow.

    :param key: Unique identifier for the state.
    :param func: Callable function that defines the action to take in this state.
    :param system_prompt: The system prompt to be sent to the model.
    :param temperature: The temperature setting for the model's response generation.
    :param transitions: A dictionary mapping possible user inputs to the next state.
    :param response_model: The Pydantic model that will parse the AI's response, if provided.
    :param pre_process_chat: Optional callable for pre-processing the chat input before running the state function.
    :param pre_process_system_prompt: Optional callable for pre-processing the system prompt before it is sent.

    :type key: str
    :type func: Callable
    :type system_prompt: str
    :type temperature: float
    :type transitions: dict[str, str]
    :type response_model: Type[BaseModel], optional
    :type pre_process_chat: Callable, optional
    :type pre_process_system_prompt: Callable, optional

    .. note:: The `transitions` dictionary should map input keys to corresponding state keys for proper FSM flow.
    """

    key: str
    func: Callable
    system_prompt: str
    temperature: float
    transitions: dict[str, str]
    response_model: Optional[Type[BaseModel]]
    pre_process_chat: Optional[Callable]
    pre_process_system_prompt: Optional[Callable]


class DefaultResponse(BaseModel):
    """
    Default response model for AI output.

    :param content: The content of the response from the AI model.

    :type content: str

    .. note:: This model can be extended or replaced with custom response models as needed.
    """

    content: str


class MooreRun(pydantic.BaseModel):
    """
    Represents the outcome of a single run/step in the FSM.

    :param state: The current state key of the FSM.
    :param chat_history: A list of dictionaries representing the history of the conversation.
    :param context_data: Contextual data relevant to the FSM run.
    :param response_raw: The raw response from the AI model.
    :param response: The processed response, potentially modeled by `response_model`.

    :type state: str
    :type chat_history: list[dict]
    :type context_data: dict[str, Any]
    :type response_raw: dict
    :type response: Any

    .. note:: The `response` attribute may be of any type, depending on the response model used.
    """

    state: str
    chat_history: list[dict]
    context_data: dict[str, Any]
    response_raw: dict
    response: Any


class StateMachineError(Exception):
    """
    Custom exception for errors within the FSM.

    .. note:: Raise this exception to indicate errors specific to FSM operations.
    """

    pass
