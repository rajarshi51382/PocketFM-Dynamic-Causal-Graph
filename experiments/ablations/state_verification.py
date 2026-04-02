from .base import AblationExperiment
from core.data_structures import CharacterState
from reasoning.state_update import propagate_state_updates
from generation.dialogue_generation import build_generation_prompt, generate_response
from reasoning.verifier import verify_dialogue
from core import llm_client


class StateVerificationExperiment(AblationExperiment):
    """
    Variant (D): Structured state with post-generation verification, no planning.

    Corresponds to paper §6 variant (D): State + Verification.
    """

    def _initialize_state(self) -> CharacterState:
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        1. Pre-turn state update (Update Phase)
        2. Dialogue generation conditioned on state (Act Phase)
        3. Post-hoc verification (compensatory transaction on failure)
        """
        # Phase 1: Update State
        event = self.build_event_frame(user_message, confidence=0.85)
        propagate_state_updates(self.state, event)

        # Phase 2: Act Phase — direct generation
        prompt = build_generation_prompt(self.state, user_message)
        response = generate_response(prompt)

        # Phase 3: Post-hoc verification
        is_valid, violations = verify_dialogue(response, self.state)

        if not is_valid:
            verbal_violations = "; ".join(violations)
            correction_prompt = (
                f"Response: {response}\n"
                f"Constraint violations: {verbal_violations}\n"
                f"Rewrite as {self.character_id}, correcting only the violations:\n"
                f"Corrected:"
            )
            response = llm_client.generate_text(correction_prompt) or f"[Correction required: {verbal_violations}]"

        response = response or "..."
        self._record_turn(user_message, response)
        return response

