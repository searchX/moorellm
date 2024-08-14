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
async def test_simple_transitions(
    fsm: MooreFSM,
    openai_client: openai.AsyncOpenAI,
    set_openai_response,
    openai_mock: openai_responses.OpenAIMock,
):
    AI_RESPONSE_SUCCESS_TURNON = "The light is turned on."
    AI_RESPONSE_SUCCESS_TURNOFF = "The light is turned off."
    AI_RESPONSE_IDLE = "I am idle, tell me to turn on or off the light."
    AI_RESPONSE_IDLE_ON = "I am idle, tell me when to turn off the light."

    @fsm.state(
        state_key="START",
        system_prompt="You are on/off light switching AI",
        transitions={
            "STATE_ON": "If user says to turn on light",
        },
    )
    async def start_state(fsm: MooreFSM, response: str, will_transition: bool):
        if not will_transition:
            assert response == AI_RESPONSE_IDLE
        if will_transition and fsm.get_next_state() == "STATE_ON":
            assert response == AI_RESPONSE_SUCCESS_TURNON

        return response

    @fsm.state(
        state_key="STATE_ON",
        system_prompt="Turning on the light...",
        transitions={"START": "If user says to turn off light"},
    )
    async def state_on(fsm: MooreFSM, response: str, will_transition: bool):
        if will_transition and fsm.get_next_state() == "START":
            assert response == AI_RESPONSE_SUCCESS_TURNOFF
        else:
            assert response == AI_RESPONSE_IDLE_ON

    set_openai_response(
        openai_mock, DefaultResponse(content=AI_RESPONSE_IDLE), next_state="START"
    )

    # Run the FSM
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "START"
    assert run_state.response == AI_RESPONSE_IDLE

    # Now test the transitions
    set_openai_response(
        openai_mock,
        DefaultResponse(content=AI_RESPONSE_SUCCESS_TURNON),
        next_state="STATE_ON",
    )
    run_state: MooreRun = await fsm.run(openai_client, user_input="Turn on the light")
    assert run_state.state == "STATE_ON"
    assert run_state.response == AI_RESPONSE_SUCCESS_TURNON

    set_openai_response(
        openai_mock, DefaultResponse(content=AI_RESPONSE_IDLE_ON), next_state="STATE_ON"
    )
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "STATE_ON"
    assert run_state.response == AI_RESPONSE_IDLE_ON

    set_openai_response(
        openai_mock,
        DefaultResponse(content=AI_RESPONSE_SUCCESS_TURNOFF),
        next_state="START",
    )

    run_state: MooreRun = await fsm.run(openai_client, user_input="Turn off the light")
    assert run_state.state == "START"
    assert run_state.response == AI_RESPONSE_SUCCESS_TURNOFF
