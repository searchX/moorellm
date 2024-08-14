import json
from typing import Callable, Dict, Any, Literal, Optional, Type, Union
from pydantic import BaseModel, ValidationError, create_model

from moorellm.models import MooreState, DefaultResponse


def _create_response_model(
    current_state_model: Union[Type[BaseModel], None],
    transitions: dict[str, str],
    default_state: str,
) -> Type[BaseModel]:
    """Create a response model based on the current state model and transitions, this will be used for structured_response openai param."""

    # Extract the transition keys as a tuple for the Literal type
    transition_keys = tuple([default_state] + list(transitions.keys()))

    next_state_key_type = Literal.__getitem__(transition_keys)

    if not current_state_model:
        current_state_model = DefaultResponse

    # Dynamically create the model with response and next_state_key fields
    return create_model(
        "EnclosedResponse",
        response=(current_state_model, ...),
        next_state_key=(next_state_key_type, ...),
    )


def _add_transitions(system_prompt: str, moore_state: MooreState) -> str:
    """Add transitions to the system prompt."""
    system_prompt += f"\n\nYou are currently in {moore_state.key} and based on user input, you can transition to the following states (with conditions defined):"
    for key, value in moore_state.transitions.items():
        system_prompt += f"\n- {key}: {value}"

    system_prompt += "\n\nIn response add the state you want to transition to.. (or leave blank to stay in the current state)"
    return system_prompt


def wrap_into_json_response(data: BaseModel, next_state: str) -> BaseModel:
    dict_res = {"response": data.model_dump(), "next_state_key": next_state}

    return json.dumps(dict_res)
