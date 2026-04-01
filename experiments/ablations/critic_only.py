from .base import AblationExperiment
from core.data_structures import CharacterState
from core import llm_client
from reasoning.verifier import verify_dialogue

class CriticOnlyExperiment(AblationExperiment):
    """
    Variant (F): Context-only baseline with post-hoc verification logic.
    """
    
    def _initialize_state(self) -> CharacterState:
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        Baseline generation + Verifier applied to the raw output.
        Note: Verification in this variant will use the state snapshot, 
        but the state is NOT updated before generation.
        """
        # Baseline Prompt (No state context)
        baseline_prompt = f"Roleplay as {self.character_id}.\nUser says: {user_message}\nResponse:"
        response = llm_client.generate_text(baseline_prompt) or "..."
        
        # Verify the raw response against the character's static state
        is_valid, _ = verify_dialogue(response, self.state)
        
        if not is_valid:
            return "[Critic-Filtered Baseline Response]"
            
        return response
