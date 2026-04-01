import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from abc import ABC, abstractmethod
from core.data_structures import CharacterState, EventFrame

class AblationExperiment(ABC):
    """
    Base class for ablation experiments.
    Defines the interface for running a dialogue turn in different configurations.
    """
    
    def __init__(self, character_id: str):
        self.character_id = character_id
        self.state = self._initialize_state()

    @abstractmethod
    def _initialize_state(self) -> CharacterState:
        """Initialize the character state for this experiment."""
        pass

    @abstractmethod
    def run_turn(self, user_message: str) -> str:
        """Execute one dialogue turn and return the character response."""
        pass

    def get_state_snapshot(self):
        """Return a snapshot of the current state for evaluation."""
        return self.state.to_dict()
