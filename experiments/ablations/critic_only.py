from .base import AblationExperiment
from core.data_structures import CharacterState
from core import llm_client
from reasoning.verifier import verify_dialogue


class CriticOnlyExperiment(AblationExperiment):
    """
    Variant (F): Context-only baseline + post-hoc verification (no DCCG state updates).

    Corresponds to paper §6 variant (F): Critic-Only Control.
    Tests whether improvements come solely from critique rather than structured state.
    The verifier receives dialogue history and explicit constraints
    but no structured belief graph.
    """

    def _initialize_state(self) -> CharacterState:
        # State is STATIC — it is never updated during generation.
        # The verifier checks against this fixed initial profile.
        return CharacterState(character_id=self.character_id)

    def run_turn(self, user_message: str) -> str:
        """
        Context-only generation (no state update) + post-hoc critic filter.
        State is intentionally frozen to isolate critic contribution.
        """
        # Build history context (context window only, no structured state)
        history_lines = []
        for turn in self.conversation_history[-10:]:
            history_lines.append(f"User: {turn['user']}")
            history_lines.append(f"{self.character_id}: {turn['response']}")
        history_section = "\n".join(history_lines)

        prompt = (
            f"Roleplay as '{self.character_id}'.\n"
        )
        if history_section:
            prompt += f"History:\n{history_section}\n"
        prompt += f"User: {user_message}\nResponse:"

        response = llm_client.generate_text(prompt) or "..."

        # Verifier checks against the STATIC initial state (no updates)
        is_valid, violations = verify_dialogue(response, self.state)

        if not is_valid:
            correction_prompt = (
                f"Response: {response}\n"
                f"Violations found: {'; '.join(violations)}\n"
                f"Rewrite as {self.character_id} to fix these issues:"
            )
            response = llm_client.generate_text(correction_prompt) or "[Critic-Filtered Response]"

        self._record_turn(user_message, response)
        return response

