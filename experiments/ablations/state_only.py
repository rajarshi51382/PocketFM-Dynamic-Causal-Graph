from .base import AblationExperiment
from core.data_structures import CharacterState
from reasoning.state_update import propagate_state_updates
from generation.dialogue_generation import build_generation_prompt, generate_response


class StateOnlyExperiment(AblationExperiment):
    """
    Variant (B): Structured state (DCCG) updates but direct generation
    without planning or verification.

    Corresponds to paper §6 variant (B):
        ŷ_t ~ P(y_t | s_t, u_t)
    """

    def _initialize_state(self) -> CharacterState:
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        1. Pre-turn state update (Update Phase)
        2. Dialogue generation conditioned on updated state (Act Phase)
        No planning or verification.
        """
        # Phase 1: Update State
        event = self.build_event_frame(user_message, confidence=0.8)
        propagate_state_updates(self.state, event)

        # Phase 2: Act Phase — conditioned generation (no plan, no verify)
        prompt = build_generation_prompt(self.state, user_message)
        response = generate_response(prompt) or "..."

        self._record_turn(user_message, response)
        return response

