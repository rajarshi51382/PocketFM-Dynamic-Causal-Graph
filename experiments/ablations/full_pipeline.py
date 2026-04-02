from .base import AblationExperiment
from core.data_structures import CharacterState
from core import llm_client
from reasoning.state_update import propagate_state_updates
from generation.dialogue_generation import build_generation_prompt, generate_response
from reasoning.verifier import verify_dialogue


class FullPipelineExperiment(AblationExperiment):
    """
    Variant (E): The full DCCG pipeline (State + Planning + Verification).

    Corresponds to paper §6 variant (E): State + Planning + Verification.
    """

    def _initialize_state(self) -> CharacterState:
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        1. Pre-turn state update (Update Phase)
        2. Intermediate plan generation (Think-on-Graph / GoT)
        3. Conditioned generation with plan context (Act Phase)
        4. Post-hoc verification + self-refine loop
        """
        # Phase 1: Update State
        event = self.build_event_frame(user_message, confidence=0.9)
        propagate_state_updates(self.state, event)

        # Phase 2: Planning (intermediate plan node I_t)
        plan_prompt = (
            f"Character State: {self.state.to_dict()}\n"
            f"User message: \"{user_message}\"\n"
            f"Generate a concise dialogue plan (intent, tone, key points to address):\n"
            f"Plan:"
        )
        plan = llm_client.generate_text(plan_prompt) or "standard dialogue plan"

        # Phase 3: Act Phase — conditioned generation
        prompt = build_generation_prompt(self.state, user_message)
        prompt += f"\nDialogue Plan: {plan}"
        response = generate_response(prompt)

        # Phase 4: Verification (Global Validation Agent)
        is_valid, violations = verify_dialogue(response, self.state)

        if not is_valid:
            # SELF-REFINE loop: convert violations into verbal feedback [paper ref 6, 7]
            verbal_feedback = "; ".join(violations)
            self_refine_prompt = (
                f"Original Response: {response}\n"
                f"Detected Violations: {verbal_feedback}\n"
                f"Rewrite the response to fix these issues while staying fully in character "
                f"as {self.character_id}. Corrected Response:"
            )
            response = llm_client.generate_text(self_refine_prompt) or "[Self-Corrected Response]"

        response = response or "..."
        self._record_turn(user_message, response)
        return response

