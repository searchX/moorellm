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
async def test_config_data(
    fsm, openai_client, set_openai_response, openai_mock: openai_responses.OpenAIMock
):
    """Test that a contexts work"""

    COMMON_RESPONSE = "My name is Moore."
    CONTEXT_SET_DATA = {
        "name": "Moore",
        "age": 25,
    }

    @fsm.state(state_key="START", system_prompt="Hello, what's your name?")
    async def start_state(fsm: MooreFSM, response: str, will_transition: bool):
        assert response == COMMON_RESPONSE
        assert will_transition == False
        assert fsm.get_next_state() == "START"

        # Set the context data
        fsm.set_context_data("config", CONTEXT_SET_DATA)

    set_openai_response(
        openai_mock, DefaultResponse(content=COMMON_RESPONSE), next_state="START"
    )

    # Run the FSM
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "START"
    assert run_state.response == COMMON_RESPONSE

    # Check the context data
    assert run_state.context_data.get("config") == CONTEXT_SET_DATA
