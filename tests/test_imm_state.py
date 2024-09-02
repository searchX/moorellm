import openai_responses
from pydantic import BaseModel
import pytest
import openai
from moorellm import MooreFSM, MooreState, DefaultResponse
from moorellm.models import MooreRun, ImmediateStateChange
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
async def test_immediete_state_transition(
    fsm: MooreFSM,
    openai_client: openai.AsyncOpenAI,
    set_openai_response,
    openai_mock: openai_responses.OpenAIMock,
):
    @fsm.state(
        state_key="START",
        system_prompt="After this Change state but not immediately",
        transitions={
            "CHANGE_NOT_IMMEDIATE": "change not immediately",
            "CHANGE_IMMEDIATE": "Changing state immediately"
        },
    )
    async def start_state(fsm: MooreFSM, response: str, will_transition: bool):
        if (will_transition and fsm.get_next_state() == "CHANGE_IMMEDIATE"):
            # Logic to do urgent state change with response from next state :)
            return ImmediateStateChange(next_state="CHANGE_IMMEDIATE")

        return "I REPLIED FROM START_STATE"
    
    @fsm.state(
        state_key="CHANGE_NOT_IMMEDIATE",
        system_prompt="Change state but not immediately",
        transitions={
            "START": "Change to start state"
        },
    )
    async def change_not_immediate_state(fsm: MooreFSM, response: str, will_transition: bool):
        return "I REPLIED FROM CHANGE_NOT_IMMEDIATE_STATE"
    
    @fsm.state(
        state_key="CHANGE_IMMEDIATE",
        system_prompt="Change state immediately",
        transitions={
            "START": "Change to start state"
        },
    )
    async def change_immediate_state(fsm: MooreFSM, response: str, will_transition: bool):
        return "I REPLIED FROM CHANGE_IMMEDIATE_STATE"

    set_openai_response(
        openai_mock, DefaultResponse(content=""), next_state="CHANGE_NOT_IMMEDIATE"
    )

    # Run the FSM
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "CHANGE_NOT_IMMEDIATE"
    # First the response should be from the start state (as we made simple transition)
    assert run_state.response == "I REPLIED FROM START_STATE"

    # For sake of test, set the next state to be main again
    fsm._state = "START"

    set_openai_response(
        openai_mock, DefaultResponse(content=""), next_state="CHANGE_IMMEDIATE"
    )

    # Now test the transitions in case immediate state change
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "CHANGE_IMMEDIATE"
    # Now it should reply from the immediate state
    assert run_state.response == "I REPLIED FROM CHANGE_IMMEDIATE_STATE"