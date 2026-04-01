from .base import AblationExperiment
from core.data_structures import CharacterState, EventFrame
from reasoning.state_update import propagate_state_updates
from generation.dialogue_generation import build_generation_prompt, generate_response
from reasoning.verifier import verify_dialogue

class StateVerificationExperiment(AblationExperiment):
    """
    Variant (D): Structured state with post-generation verification, 
    but no planning.
    """
    
    def _initialize_state(self) -> CharacterState:
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        1. Pre-turn state update
        2. Dialogue generation (conditioned on state)
        3. Post-hoc verification
        """
        # Phase 1: Update State
        event = EventFrame(
            subject="user",
            entities=[self.character_id],
            action="say",
            value=user_message,
            emotional_tone="neutral",
            confidence=0.85
        )
        propagate_state_updates(self.state, event)

        # Phase 2: Act Phase
        prompt = build_generation_prompt(self.state, user_message)
        response = generate_response(prompt)
        
        # Phase 3: Verification (Post-hoc filter)
        is_valid, violations = verify_dialogue(response, self.state)
        
        if not is_valid:
            # Simple fallback on violation
            return f"[Self-Correction Required: {violations}]"
            
        return response if response else "..."
