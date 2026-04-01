from .base import AblationExperiment
from core.data_structures import CharacterState
from core import llm_client

class BaselineLMExperiment(AblationExperiment):
    """
    Variant (A): Standard context-only generation without any structured state.
    """
    
    def _initialize_state(self) -> CharacterState:
        # Minimal state, we don't use it in this variant
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        Baseline generation: Just feed the context (history + user message) 
        to the LLM without structured state injection.
        """
        # In a real experiment, we'd include conversation history here
        prompt = (
            f"You are roleplaying as {self.character_id}.\n"
            f"User says: \"{user_message}\"\n"
            f"Response:"
        )
        
        response = llm_client.generate_text(prompt)
        return response if response else "..."
