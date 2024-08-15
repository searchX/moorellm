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


# Test for simple fsm with state
@pytest.mark.asyncio
@openai_responses.mock()
async def test_jinja2(
    fsm: MooreFSM,
    openai_client: openai.AsyncOpenAI,
    set_openai_response,
    openai_mock: openai_responses.OpenAIMock,
):
    JINJA2_TEMPLATE = "Hello, {{ name }} is here."
    user_defined_variables = {"name": "John"}

    # Set variables
    fsm.set_context_data_dict(user_defined_variables)

    def assert_system_prompt(system_prompt: str, fsm: MooreFSM):
        assert system_prompt == "Hello, John is here."

    @fsm.state(
        state_key="START",
        system_prompt=JINJA2_TEMPLATE,
        pre_process_system_prompt=assert_system_prompt,
    )
    async def start_state(fsm: MooreFSM, response: str, will_transition: bool):
        pass

    set_openai_response(openai_mock, DefaultResponse(content=""), next_state="START")

    # Run the FSM
    run_state: MooreRun = await fsm.run(openai_client, user_input="Hello")
    assert run_state.state == "START"
