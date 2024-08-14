import openai
from moorellm import MooreFSM
from moorellm.models import MooreRun

LIGHT_STATE = "OFF"

# Create the FSM
fsm = MooreFSM(initial_state="START")


# Define the states
@fsm.state(
    state_key="START",
    system_prompt="You are AI light switcher, Ask user if they want to turn on the light.",
    transitions={"STATE_ON": "If user says to turn on the light"},
)
async def start_state(fsm: MooreFSM, response: str, will_transition: bool):
    """Default state when light is off

    :param fsm: The Moore FSM object
    :param response: The response from the AI model
    :param will_transition: Whether the FSM will transition to the next state, if true then the light will turn on after the transition
    :type fsm: :class:`moorellm.MooreFSM`
    :type response: str
    :type will_transition: bool

    :return: The response from the AI model (which is to be displayed to the user)
    :rtype: str
    """
    global LIGHT_STATE

    if will_transition and fsm.get_next_state() == "STATE_ON":
        LIGHT_STATE = "ON"
        print("LIGHT TURNED ON")

    return response


@fsm.state(
    state_key="STATE_ON",
    system_prompt="Turning on the light...",
    transitions={"START": "If user says to turn off the light"},
)
async def state_on(fsm: MooreFSM, response: str, will_transition: bool):
    """State when light is on, user can turn off the light

    :param fsm: The Moore FSM object
    :param response: The response from the AI model
    :param will_transition: Whether the FSM will transition to the next state, if true then the light will turn off after the transition
    :type fsm: :class:`moorellm.MooreFSM`
    :type response: str
    :type will_transition: bool

    :return: The response from the AI model (which is to be displayed to the user)
    :rtype: str
    """
    global LIGHT_STATE

    if will_transition and fsm.get_next_state() == "START":
        LIGHT_STATE = "OFF"
        print("LIGHT TURNED OFF")

    return response


async def main():
    """Example of a simple light switch FSM using MooreFSM"""
    # Create the OpenAI client
    openai_client = openai.AsyncOpenAI()
    # Simulate conversation
    while True:
        user_input = input("You: ")
        run_state: MooreRun = await fsm.run(openai_client, user_input=user_input)
        print(f"AI: {run_state.response}")
        print("CURRENT LIGHT STATE: ", LIGHT_STATE)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
