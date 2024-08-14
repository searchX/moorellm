import openai_responses
from pydantic import BaseModel
import pytest
import openai
from moorellm import MooreFSM, MooreState, DefaultResponse
from moorellm.models import MooreRun
from moorellm.utils import wrap_into_json_response


# Define the fixture
@pytest.fixture
def fsm():
    """Fixture for creating a MooreFSM instance."""
    return MooreFSM(initial_state="START")


@pytest.fixture
def openai_client():
    return openai.AsyncOpenAI(api_key="sk-fake123")


@pytest.fixture
def set_openai_response():
    def set_response(
        openai_mock: openai_responses.OpenAIMock, response: BaseModel, next_state: str
    ):
        openai_mock.beta.chat.completions.create.response = {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "content": wrap_into_json_response(response, next_state),
                        "role": "assistant",
                    },
                }
            ]
        }

    return set_response


# Test the module creation
def test_module_creation(fsm):
    """Test that the module can be created."""
    assert isinstance(fsm, MooreFSM)


# Test for simple fsm with state
@pytest.mark.asyncio
@openai_responses.mock()
async def test_single_state(
    fsm, openai_client, set_openai_response, openai_mock: openai_responses.OpenAIMock
):
    """Test that a single state can be defined."""

    COMMON_RESPONSE = "My name is Moore."

    @fsm.state(state_key="START", system_prompt="Hello, what's your name?")
    async def start_state(fsm: MooreFSM, response: str, will_transition: bool):
        assert response == COMMON_RESPONSE
        assert will_transition == False
        assert fsm.get_next_state() == "START"

    set_openai_response(
        openai_mock, DefaultResponse(content=COMMON_RESPONSE), next_state="START"
    )

    # Run the FSM
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "START"
    assert run_state.response == COMMON_RESPONSE


@pytest.mark.asyncio
@openai_responses.mock()
async def test_response_manipulation(
    fsm, openai_client, set_openai_response, openai_mock: openai_responses.OpenAIMock
):
    """Test that the response can be manipulated in a state."""

    NATURAL_RESPONSE = "My name is Moore."
    MANIPULATED_RESPONSE = "My name is Moore. What's yours?"

    @fsm.state(state_key="START", system_prompt="Hello, what's your name?")
    async def start_state(fsm: MooreFSM, response: str, will_transition: bool):
        assert response == NATURAL_RESPONSE
        assert will_transition == False
        assert fsm.get_next_state() == "START"
        return MANIPULATED_RESPONSE

    set_openai_response(
        openai_mock, DefaultResponse(content=NATURAL_RESPONSE), next_state="START"
    )

    # Run the FSM
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "START"
    assert run_state.response == MANIPULATED_RESPONSE


# Test for simple fsm with models
@pytest.mark.asyncio
@openai_responses.mock()
async def test_single_state_with_model(
    fsm, openai_client, set_openai_response, openai_mock: openai_responses.OpenAIMock
):
    """Test that response model can be generated and then given to a state."""

    class NameResponse(BaseModel):
        name: str

    EXTRACTED_NAME = "Moore"
    GREETING_PREFIX = "Hello, Welcome: "

    @fsm.state(
        state_key="START",
        system_prompt="You are an AI which extracts user's name, ask the user for their name.",
        response_model=NameResponse,
    )
    async def start_state(fsm: MooreFSM, response: NameResponse, will_transition: bool):
        assert response.name == "Moore"
        assert will_transition == False
        assert fsm.get_next_state() == "START"
        return GREETING_PREFIX + response.name

    set_openai_response(openai_mock, NameResponse(name="Moore"), next_state="START")

    # Run the FSM
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "START"
    assert run_state.response == GREETING_PREFIX + EXTRACTED_NAME
