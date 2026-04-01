from .base import AblationExperiment
from core.data_structures import CharacterState, EventFrame
from core import llm_client
from reasoning.state_update import propagate_state_updates
from generation.dialogue_generation import build_generation_prompt, generate_response
from reasoning.verifier import verify_dialogue

class FullPipelineExperiment(AblationExperiment):
    """
    Variant (E): The full DCCG pipeline (State + Planning + Verification).
    """
    
    def _initialize_state(self) -> CharacterState:
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        1. Pre-turn state update
        2. Dialogue planning (Intermediate node)
        3. Conditioned generation (with plan)
        4. Post-hoc verification
        """
        # Phase 1: Update State
        event = EventFrame(
            subject="user",
            entities=[self.character_id],
            action="say",
            value=user_message,
            emotional_tone="neutral",
            confidence=0.9
        )
        propagate_state_updates(self.state, event)

        # Phase 2: Planning
        plan_prompt = f"Plan dialogue for {self.character_id} considering {self.state.to_dict()} and input \"{user_message}\"."
        plan = llm_client.generate_text(plan_prompt) or "standard dialogue plan"

        # Phase 3: Act Phase (Conditioned Generation)
        prompt = build_generation_prompt(self.state, user_message)
        prompt += f"\nPlan Context: {plan}"
        response = generate_response(prompt)
        
        # Phase 4: Verification (Closed-loop)
        is_valid, violations = verify_dialogue(response, self.state)
        
        if not is_valid:
            # Full implementation would trigger a "Self-Refine" Loop as described in paper [6, 7]
            self_refine_prompt = f"Original Response: {response}\nViolations: {violations}\nIn-character correction:"
            return llm_client.generate_text(self_refine_prompt) or "[Self-Corrected Response]"
            
        return response if response else "..."
