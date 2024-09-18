from typing import Union
import openai
import jinja2
from pydantic import BaseModel
from moorellm.models import GuardrailResponse
import logging


# ------ Define Helper Functions ------ #
async def structured_call(
    chat_history: list,
    openai_instance: Union[openai.AzureOpenAI, openai.AsyncAzureOpenAI],
    response_model: type[BaseModel],
    model_name: str = "gpt-4o-2024-08-06",
    **kwargs,
):
    completion: openai.types.Completion = (
        await openai_instance.beta.chat.completions.parse(
            model=model_name,
            messages=chat_history,
            response_format=response_model,
        )
    )

    message = completion.choices[0].message
    if message.parsed:
        message: response_model = message.parsed
        return message
    else:
        print(message.refusal)
        raise Exception(
            "Failed to get structured_call, please try again.." + message.refusal
        )


# Guardrail transition function for moorellm
async def guardrail_transition(
    condition: str,
    state_from: str,
    state_to: str,
    chat_history: list[dict],
    openai_instance: Union[openai.AzureOpenAI, openai.AsyncAzureOpenAI],
):
    """Guardrail transition function for moorellm, this function will check if the transition is valid or not.

    :param condition: The condition for the transition
    :param state_from: The state from which the transition is happening
    :param state_to: The state to which the transition is happening
    :param chat_history: The chat history upto this point
    :param openai_instance: The openai instance to use
    :type condition: str
    :type state_from: str
    :type state_to: str
    :type chat_history: list[dict]
    :type openai_instance: Union[openai.AzureOpenAI, openai.AsyncAzureOpenAI]
    :return: The response from the guardrail
    :rtype: GuardrailResponse
    """
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

    response: GuardrailResponse = await structured_call(
        chat_history, openai_instance, GuardrailResponse
    )
    return response


async def llm_judge_response(
    response: str,
    good_or_bad_desc: str,
    openai_instance: Union[openai.AzureOpenAI, openai.AsyncAzureOpenAI],
    max_voting_runs: int = 1,
) -> bool:
    """Use LLM to judge if the response is good or bad as compared to the original response, using voting for n times.

    :param response: The response to judge
    :param good_or_bad_desc: The description of good or bad response
    :param openai_instance: The openai instance to use
    :param max_voting_runs: The number of times to vote (must be odd and greater than 0), defaults to 1
    :type response: str
    :type good_or_bad_desc: str
    :type openai_instance: Union[openai.AzureOpenAI, openai.AsyncAzureOpenAI]
    :type max_voting_runs: int, optional

    :return: The judged response (true if good, false if bad)
    :rtype: bool
    """
    if max_voting_runs % 2 == 0 or max_voting_runs <= 0:
        raise ValueError("max_voting_runs must be odd and greater than 0")
    # Using voting for n times and make LLM judge if our response is good or bad as compared to the original response
    good_votes = 0
    bad_votes = 0

    logger = logging.getLogger("moorellm")

    class JudgeResponse(BaseModel):
        is_response_good: bool
        thinking_steps: str

    system_prompt = f"""You are an AI Response judge system responsible for judging if the response is good or bad as compared to the how it is expected to be. \
                        Given the response and the description of good or bad response, you need to judge if the response is good or bad, Think clearly and provide the judgement."""

    user_content = (
        f"""Response: {response}, Good or Bad Description: {good_or_bad_desc}"""
    )

    for i in range(max_voting_runs):
        response: JudgeResponse = await structured_call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            openai_instance,
            JudgeResponse,
        )

        logger.debug(
            f"{i} - Response: {response.is_response_good}, Thinking Steps: {response.thinking_steps}"
        )
        if response.is_response_good:
            good_votes += 1
        else:
            bad_votes += 1

    return good_votes > bad_votes
