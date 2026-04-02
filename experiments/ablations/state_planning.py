from .base import AblationExperiment
from core.data_structures import CharacterState
from core import llm_client
from reasoning.state_update import propagate_state_updates
from generation.dialogue_generation import build_generation_prompt, generate_response


class StatePlanningExperiment(AblationExperiment):
    """
    Variant (C): Structured state with an intermediate planning step, no verification.

    Corresponds to paper §6 variant (C): State + Planning.
    """

    def _initialize_state(self) -> CharacterState:
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        1. Pre-turn state update (Update Phase)
        2. Intermediate plan node I_t generation
        3. Conditioned generation (Act Phase, no verification)
        """
        # Phase 1: Update State
        event = self.build_event_frame(user_message, confidence=0.9)
        propagate_state_updates(self.state, event)

        # Phase 2: Intermediate plan node
        plan_prompt = (
            f"Character State: {self.state.to_dict()}\n"
            f"User message: \"{user_message}\"\n"
            f"Formulate a concise dialogue plan (intent, tone, strategy).\n"
            f"Plan:"
        )
        plan = llm_client.generate_text(plan_prompt) or "neutral response"

        # Phase 3: Generation conditioned on state + plan
        prompt = build_generation_prompt(self.state, user_message)
        prompt += f"\nPlan: {plan}"
        response = generate_response(prompt) or "..."

        self._record_turn(user_message, response)
        return response

