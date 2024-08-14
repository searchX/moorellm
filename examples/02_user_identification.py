import openai
from pydantic import BaseModel
from moorellm import MooreFSM
from moorellm.models import MooreRun

# Create the FSM
fsm = MooreFSM(initial_state="START", end_state="IDENTIFIED")


class UserIdentificationResponse(BaseModel):
    content: str
    user_name: str
    phone_number: str


# Define the states
@fsm.state(
    state_key="START",
    system_prompt="You are user identification bot, You have to identify the user by asking their name and phone number.",
    response_model=UserIdentificationResponse,
    transitions={"IDENTIFIED": "If user provides their name and phone number"},
)
async def start_state(
    fsm: MooreFSM, response: UserIdentificationResponse, will_transition: bool
):
    if will_transition and fsm.get_next_state() == "IDENTIFIED":
        # Set into context
        fsm.set_context_data(
            "verified_user",
            {
                "user_name": response.user_name,
                "phone_number": response.phone_number,
            },
        )
        return "Thanks for Identifying yourself."

    return response.content


async def main():
    # Create the OpenAI client
    openai_client = openai.AsyncOpenAI()
    # Simulate conversation
    while fsm.is_completed() is False:
        user_input = input("You: ")
        run_state: MooreRun = await fsm.run(openai_client, user_input=user_input)
        print(f"AI: {run_state.response}")

    # Get the context data
    context_data = fsm.get_context_data("verified_user")
    print(f"User Identified: {context_data}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
