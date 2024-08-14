# MooreLLM - FSM based Agentic Approach for AI Applications

Welcome to the MooreLLM project! This project is a FSM based approach for creating agentic applications..

## Why MooreLLM?
The current approach (as i have observed) for creating LLM agents are simply *giving prompts and some tools*, which the LLM can use to generate the desired output. This approach is not very flexible as it may hallucinate or call wrong tools, MooreLLM is an attempt at creating a very controlled LLM agent which has set of states and transitions, and can only perform actions that are allowed in the current state.

## An example to illustrate the need for MooreLLM
For example, imagine a call-center chatbot, which will converse according to current state, ie if user has just started the chat, the bot will ask for the user's name, and then greet the user.. then **only when** user has provided the name, the bot will ask for the user's query.. and so on..

ie the conversation pattern will follow a specific flow, and the bot SHOULD NEVER ask for the query before the user has provided the name.

### The current approach (Traditional Approach)
Now imagine a function based approach, where the bot can ask for the name, greet the user, and ask for the query, all at the same time.. this is not a very good approach as the bot can ask for the query even before the user has provided the name.. this can lead to a bad user experience and potentially uncontrolled behavior.

<img src="https://raw.githubusercontent.com/searchX/moorellm/main/images/traditional_approach.png" height="512">

Current approach of mitigating such issues is instructing in system_prompt about the set order/sequence, but this is error prone and not very flexible.

### What we Propose (MooreLLM Approach)
Create states for each of the conversation steps, and transitions between them.. this way the bot can only ask for the query when the user has provided the name.. this is a very controlled approach and can lead to a better user experience.

ie now I will have seperate system_prompt for both greeting and asking for the query, so *until I am in greet user state, I will only use the system_prompt for greeting, and once I have moved to ask_for_query state, I will only use the system_prompt for asking for the query*.

<!-- ![MooreLLM Approach]() -->
<img src="https://raw.githubusercontent.com/searchX/moorellm/main/images/moore_approach.png" height="512">

## What is an Agentic AI Application?
It is an application that can act on its own, without the need for human intervention. It can make decisions, take actions, and interact with the environment.. for example
- A chatbot that can answer questions
- A robot that can perform tasks
- A game character that can play a game

## What is an FSM?
A Finite State Machine (FSM) is a mathematical model of computation. It is a set of states, transitions, and actions. The machine can be in only one state at a time, and it can change from one state to another in response to some inputs. FSMs are widely used in many applications, such as games, control systems, and natural language processing.

example of a FSM:
![FSM Example](https://raw.githubusercontent.com/searchX/moorellm/main/images/01_bulb_state_diagram.png)

## How to Design an Agentic AI Application using MooreLLM?
To use MooreLLM, you need to define the states and transitions! Each state will have base system prompts, and transitions will have the conditions to move from one state to another, for each state you can even parse complex generated output or make use of context variables. (We will see this in the examples)

## Installation
Install the ``moorellm`` package with [pip](https://pypi.org/project/moorellm)

```console
$ pip install moorellm
```

Or install the latest package directly from github

```console
$ pip install git+https://github.com/searchX/moorellm
```
## Example Usage

### A Light Bulb Chatbot

Create FSM
```python
import openai
from moorellm import MooreFSM
from moorellm.models import MooreRun

# Create the FSM
fsm = MooreFSM(initial_state="START")
LIGHT_STATE = "OFF" # Default state
```

Define default state `START`
```python
# Define the states
@fsm.state(
    state_key="START",
    system_prompt="You are AI light switcher, Ask user if they want to turn on the light.",
    transitions={"STATE_ON": "If user says to turn on the light"},
)
async def start_state(fsm: MooreFSM, response: str, will_transition: bool):
    global LIGHT_STATE

    if will_transition and fsm.get_next_state() == "STATE_ON":
        LIGHT_STATE = "ON"
        print("LIGHT TURNED ON")

    return response
```

Define state `STATE_ON`
```python
@fsm.state(
    state_key="STATE_ON",
    system_prompt="Turning on the light...",
    transitions={"START": "If user says to turn off the light"},
)
async def state_on(fsm: MooreFSM, response: str, will_transition: bool):
    global LIGHT_STATE

    if will_transition and fsm.get_next_state() == "START":
        LIGHT_STATE = "OFF"
        print("LIGHT TURNED OFF")

    return response
```

Run the FSM
```python
async def main():
    # Create the OpenAI client
    openai_client = openai.AsyncOpenAI()
    # Simulate conversation
    while True:
        user_input = input("You: ")
        run_state: MooreRun = await fsm.run(openai_client, user_input=user_input)
        print(f"AI: {run_state.response}")
        print("CURRENT LIGHT STATE: ", LIGHT_STATE)

import asyncio
asyncio.run(main())
```

States:
- START: The initial state where the AI asks the user if they want to turn on the light.
- STATE_ON: The state where the AI turns, only avaliable transition at this state is to turn off the light.

Notice the transitions, for state `start` we defined only one transition to `STATE_ON`, and for state `STATE_ON` we defined only one transition to `START`, this way the AI can only move from `start` to `STATE_ON` and then from `STATE_ON` to `start` and not any other state.

Note: Default transitions are always available, so if the user input does not match any of the transitions, the AI will move to the same state (ie loop back).

### User identification
```python
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
```
```python
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
```
Call it with
```python
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
```

Sample Output
```console
You: Hi
AI: Hello there! Could you please provide your name and phone number so I can identify you?
You: My name is Harishankar
AI: Thanks for providing your name. Could you please provide your phone number?
You: My phone number is 1234567890
AI: Thanks for Identifying yourself.
User Identified: {'user_name': 'Harishankar', 'phone_number': '1234567890'}
```

More examples can be found in the [examples](examples) directory.

## Notes
- The MooreLLM is a very basic implementation, and can be extended to support more complex use-cases.
- The FSM is designed to be used with OpenAI's GPT-4o+ (as it uses structured responses), but can be used with any other model as well (with some modification).
- Possibility of bad return json is non-existent, this is because we are using the latest structured responses from OpenAI, and the MooreLLM is designed to handle only structured responses.

## Contributing
MooreLLM at its current state is a very basic implementation, and very opinionated.. I would love to see more contributions and ideas on how to make it more flexible and useful for a wider range of applications, any PRs are welcome!# moorellm
