from .base import AblationExperiment
from core.data_structures import CharacterState
from core import llm_client


class BaselineLMExperiment(AblationExperiment):
    """
    Variant (A): Standard context-only generation without structured state,
    planning, or verification.

    Corresponds to paper §6 variant (A): Baseline LM.
    Character persona is implied only through the prompt — no persistent state.
    """

    def _initialize_state(self) -> CharacterState:
        # Minimal placeholder — not used during generation
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        Context-only baseline: feed conversation history + user message
        to the LLM with no structured state injection.
        """
        # Build dialogue history section (context window only)
        history_lines = []
        for turn in self.conversation_history[-10:]:  # cap at 10 turns
            history_lines.append(f"User: {turn['user']}")
            history_lines.append(f"{self.character_id}: {turn['response']}")
        history_section = "\n".join(history_lines)

        prompt = (
            f"You are roleplaying as the character '{self.character_id}'.\n"
            f"Maintain a consistent personality throughout.\n"
        )
        if history_section:
            prompt += f"\nConversation so far:\n{history_section}\n"
        prompt += f"\nUser: \"{user_message}\"\n{self.character_id}:"

        response = llm_client.generate_text(prompt) or "..."
        self._record_turn(user_message, response)
        return response

