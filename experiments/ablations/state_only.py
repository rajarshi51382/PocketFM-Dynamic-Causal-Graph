from .base import AblationExperiment
from core.data_structures import CharacterState, EventFrame
from reasoning.state_update import propagate_state_updates
from generation.dialogue_generation import build_generation_prompt, generate_response

class StateOnlyExperiment(AblationExperiment):
    """
    Variant (B): Structured state (DCCG) updates but direct generation 
    without planning or verification.
    """
    
    def _initialize_state(self) -> CharacterState:
        # Load character state from current source definitions
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        1. Pre-turn state update
        2. Dialogue generation (conditioned on state)
        3. Simple turn completion (no planning/verification)
        """
        # Simplified extraction logic
        event = EventFrame(
            subject=self.character_id,
            entities=[],
            action="receive",
            value=user_message,
            emotional_tone="neutral",
            confidence=0.8
        )
        
        # Phase 1: Update State
        propagate_state_updates(self.state, event)
        
        # Phase 2: Act Phase (Conditioned Generation)
        prompt = build_generation_prompt(self.state, user_message)
        response = generate_response(prompt)
        
        return response if response else "..."
