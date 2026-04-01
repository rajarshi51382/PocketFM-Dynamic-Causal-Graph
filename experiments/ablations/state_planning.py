from .base import AblationExperiment
from core.data_structures import CharacterState, EventFrame
from core import llm_client
from reasoning.state_update import propagate_state_updates
from generation.dialogue_generation import build_generation_prompt, generate_response

class StatePlanningExperiment(AblationExperiment):
    """
    Variant (C): Structured state with an intermediate planning step.
    """
    
    def _initialize_state(self) -> CharacterState:
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        1. Pre-turn state update
        2. Dialogue planning (Intermediate node)
        3. Conditioned generation
        """
        # Phase 1: Update State
        # In a real setup, we'd use event extraction here
        event = EventFrame(
            subject="user",
            entities=[self.character_id],
            action="say",
            value=user_message,
            emotional_tone="neutral",
            confidence=0.9
        )
        propagate_state_updates(self.state, event)

        # Intermediate Plan Phase
        plan_prompt = (
            f"Character State: {self.state.to_dict()}\n"
            f"User message: \"{user_message}\"\n"
            f"Goal: Formulate a dialogue plan (intent, tone, strategy)."
            f"Plan:"
        )
        plan = llm_client.generate_text(plan_prompt) or "neutral response"
        
        # Generation with Plan
        prompt = build_generation_prompt(self.state, user_message)
        prompt += f"\nPlan: {plan}"
        
        response = generate_response(prompt)
        return response if response else "..."
