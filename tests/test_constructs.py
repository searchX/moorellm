import pytest
from moorellm import MooreFSM


@pytest.fixture
def fsm():
    """Fixture for creating a MooreFSM instance."""
    return MooreFSM(initial_state="START")


def test_module_creation(fsm):
    """Test that the module can be created."""
    assert isinstance(fsm, MooreFSM)


def test_reset(fsm):
    """Test that the FSM can be reset."""
    fsm.set_next_state("NEXT")
    fsm.reset()
    assert fsm.get_current_state() == "START"


def test_get_next_state(fsm):
    """Test that the next state can be retrieved."""
    fsm.set_next_state("NEXT")
    assert fsm.get_next_state() == "NEXT"


def test_get_current_state(fsm):
    """Test that the current state can be retrieved."""
    assert fsm.get_current_state() == "START"
