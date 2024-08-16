from typing import Union
import openai
import jinja2
from pydantic import BaseModel
from moorellm.models import GuardrailResponse

# ------ Define Helper Functions ------ #
async def structured_call(chat_history: list, openai_instance: Union[openai.AzureOpenAI, openai.AsyncAzureOpenAI], response_model: type[BaseModel], model_name: str = "gpt-4o-2024-08-06", **kwargs):
    completion: openai.types.Completion = await openai_instance.beta.chat.completions.parse(
        model=model_name,
        messages=chat_history,
        response_format=response_model,
    )

    message = completion.choices[0].message
    if message.parsed:
        message: response_model = message.parsed
        return message
    else:
        print(message.refusal)
        raise Exception("Failed to get structured_call, please try again.." + message.refusal)

# Guardrail transition function for moorellm
async def guardrail_transition(condition: str, state_from: str, state_to: str, chat_history: list[dict], openai_instance: Union[openai.AzureOpenAI, openai.AsyncAzureOpenAI]):
    system_prompt = f"""You are an AI states management system responsible for checking if transitions
                        from one state to another are valid, You observed that a state transition was requested
                        from state {state_from} to state {state_to}, Use user conversation upto this point to verify
                        if this transition is valid or not.

                        The condition for this transition is: {condition}"""

    chat_history_copy = chat_history.copy()
    # Iterate through the chat history and remove the system prompts
    for chat in chat_history_copy:
        if chat["role"] == "system":
            chat_history.remove(chat)
    # Add the system prompt
    chat_history.append({"role": "system", "content": system_prompt})

    response: GuardrailResponse = await structured_call(chat_history, openai_instance, GuardrailResponse)
    return response