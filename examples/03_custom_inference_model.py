import openai
from pydantic import BaseModel
from moorellm import MooreFSM
from moorellm.models import MooreRun
import asyncio

# Create the FSM
fsm = MooreFSM(initial_state="TITLE", end_state="END")


# Let's use custom llm model to generate this :)
from cerebras.cloud.sdk import AsyncCerebras, CerebrasError
import instructor
from typing import Type, List, Union
from pydantic import BaseModel

class PoemTitleResponse(BaseModel):
    content: str
    title: str

class PoemToneResponse(BaseModel):
    content: str
    tone: str

class PoemResponse(BaseModel):
    content: str
    poem: str


async def _cerebras_get_completion(
    cerebras_instance: Union[AsyncCerebras],
    chat_history: List[dict],
    response_model: Type[BaseModel],
    llm_model: str,
):
    """
    Get a completion using the Cerebras SDK.

    Args:
        cerebras_instance: An instance of async Cerebras
        chat_history: The conversation history as a list of dictionaries.
        response_model: The Pydantic model to parse the response into.
        llm_model: The model name for inference.

    Returns:
        Parsed response as a Pydantic model instance dumped to a dictionary.

    Raises:
        CerebrasSDKError: If there's an issue with the response or the parsing.
    """
    try:        
        # Create chat completion asynchronously
        completion = await cerebras_instance.chat.completions.create(
            model=llm_model,
            messages=chat_history,
            response_model=response_model,
        )
        
        if not completion:
            raise CerebrasError("No completion returned from the Cerebras client.")

        return completion.model_dump()  # The response is already parsed into the specified model

    except Exception as e:
        raise CerebrasError(f"Error in fetching or parsing the completion: {str(e)}")


fsm.override_get_completion(_cerebras_get_completion)

# Define the states
@fsm.state(
    state_key="TITLE",
    system_prompt="You are a poem designer bot. Ask user for initial title and make changes/suggestions based on the tone, only move when user agrees that title looks good or they are done with it, default title is 'N/A'",
    response_model=PoemTitleResponse,
    transitions={"TONE": "If user provides a title"},
)
async def title_state(
    fsm: MooreFSM, response: PoemTitleResponse, will_transition: bool
):
    if will_transition and fsm.get_next_state() == "TONE":
        fsm.set_context_data("poem_title", response.title)
        return "Great! Now, please provide the tone for the poem."

    return response.content + "\n\n" + "Current Title: " + response.title

@fsm.state(
    state_key="TONE",
    system_prompt="Ask user for tone of the poem, only move when user agrees that tone looks good or they are done with it, default tone is 'Neutral'",
    response_model=PoemToneResponse,
    transitions={"POEM": "If user provides a tone"},
)
async def tone_state(
    fsm: MooreFSM, response: PoemToneResponse, will_transition: bool
):
    if will_transition and fsm.get_next_state() == "POEM":
        fsm.set_context_data("poem_tone", response.tone)
        return "Awesome! Now, let's create the poem."

    return response.content + "\n\n" + "Current Tone: " + response.tone

@fsm.state(
    state_key="POEM",
    system_prompt="Based on the title and tone, generate a poem. If user wants to make changes, then make changes and only move when user agrees that poem looks good or they are done with it. Title: {{ poem_title }}, Tone: {{ poem_tone }}, default poem return N/A",
    response_model=PoemResponse,
    transitions={
        "END": "If user is done with the poem",
    },
)
async def poem_state(
    fsm: MooreFSM, response: PoemResponse, will_transition: bool
):
    if will_transition:
        fsm.set_context_data("poem", response.poem)
        return f"Here is your poem:\n\n{response.poem}"

    return response.content + "\n\n" + f"Current Poem:\n\n{response.poem}"

fsm._states["END"] = None

async def main():
    # Create the OpenAI client
    cerebras_client = AsyncCerebras()
    client = instructor.from_cerebras(cerebras_client)

    # Simulate conversation
    while fsm.is_completed() is False:
        user_input = input("You: ")
        run_state: MooreRun = await fsm.run(client, user_input=user_input, model="llama3.1-70b")
        print(f"AI: {run_state.response}")

    # Get the context data
    poem_title = fsm.get_context_data("poem_title")
    poem_tone = fsm.get_context_data("poem_tone")
    poem = fsm.get_context_data("poem")
    # Print the final poem
    # Make a seperator here
    print("--------------------------------------------------")
    print(f"Poem Title: {poem_title}")
    print(f"Poem Tone: {poem_tone}")
    print(f"Poem: {poem}")
    print("--------------------------------------------------")

if __name__ == "__main__":

    asyncio.run(main())