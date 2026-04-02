import sys
import os
import copy
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from abc import ABC, abstractmethod
from core.data_structures import CharacterState, EventFrame


class AblationExperiment(ABC):
    """
    Base class for ablation experiments.
    Defines the interface for running a dialogue turn in different configurations.

    Each subclass implements one of the six ablation variants (A-F) described
    in the DCCG paper §6 (Ablation Study: Isolating State vs. Critic Effects).
    """

    def __init__(self, character_id: str):
        self.character_id = character_id
        self.state = self._initialize_state()
        # Full conversation history for metric computation
        self.conversation_history: List[Dict[str, Any]] = []
        # State snapshots (one per turn) for drift metrics
        self.state_snapshots: List[Dict[str, Any]] = []

    @abstractmethod
    def _initialize_state(self) -> CharacterState:
        """Initialize the character state for this experiment."""
        pass

    @abstractmethod
    def run_turn(self, user_message: str) -> str:
        """Execute one dialogue turn and return the character response."""
        pass

    def _record_turn(self, user_message: str, response: str) -> None:
        """Save turn data + state snapshot for downstream metric computation."""
        self.conversation_history.append({
            "turn": len(self.conversation_history),
            "user": user_message,
            "response": response,
        })
        self.state_snapshots.append(copy.deepcopy(self.state.to_dict()))

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the current state for evaluation."""
        return self.state.to_dict()

    def reset(self) -> None:
        """Reset the experiment to its initial state."""
        self.state = self._initialize_state()
        self.conversation_history = []
        self.state_snapshots = []

    def build_event_frame(
        self,
        user_message: str,
        propositions: list[str] | None = None,
        emotional_tone: str = "neutral",
        confidence: float = 0.85,
    ) -> EventFrame:
        """
        Utility for constructing an EventFrame from user input.
        Ablation subclasses call this instead of instantiating EventFrame directly.
        """
        turn_idx = len(self.conversation_history)
        return EventFrame(
            propositions=propositions or [],
            entities=[self.character_id],
            speaker="user",
            emotional_tone=emotional_tone,
            confidence=confidence,
            turn_index=turn_idx,
            source_text=user_message,
        )
