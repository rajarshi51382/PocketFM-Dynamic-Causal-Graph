"""
Stability Scenarios (No Character Change Required).

Paper §5.1:
    Extended dialogue with low-surprisal, low-intensity events (σ(e_t) ≈ 1).
    Core traits SHOULD remain stable.
    Used to measure UDR (Unjustified Drift Rate).

Each scenario is a sequence of ScenarioTurns. All messages are designed
to be routine — small talk, clarifying questions, status checks — that
should NOT trigger meaningful personality drift.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ScenarioTurn:
    """
    A single dialogue turn in an experiment scenario.

    Attributes
    ----------
    user_message : str
        The user's input for this turn.
    expected_sigma : float
        Approximate σ value the scenario designer expects (for reference/validation).
    annotation : str
        Short description of what the turn tests.
    forced_emotional_tone : str
        Optional tone label to override extraction (for controlled tests).
    """
    user_message: str
    expected_sigma: float = 1.0
    annotation: str = ""
    forced_emotional_tone: str = "neutral"


# ---------------------------------------------------------------------------
# Scenario 1: Village Storyteller (20 turns, routine chit-chat)
# ---------------------------------------------------------------------------
VILLAGE_STORYTELLER: List[ScenarioTurn] = [
    ScenarioTurn(
        user_message="Good morning! How are you doing today?",
        expected_sigma=1.0, annotation="Routine greeting",
    ),
    ScenarioTurn(
        user_message="What kind of stories do you like to tell?",
        expected_sigma=1.0, annotation="Routine open question",
    ),
    ScenarioTurn(
        user_message="Have you always lived in this village?",
        expected_sigma=1.05, annotation="Biographical question — mild curiosity",
    ),
    ScenarioTurn(
        user_message="Do you have any brothers or sisters?",
        expected_sigma=1.0, annotation="Routine biographical",
    ),
    ScenarioTurn(
        user_message="What's your favourite season?",
        expected_sigma=1.0, annotation="Small talk",
    ),
    ScenarioTurn(
        user_message="Do you enjoy the market days?",
        expected_sigma=1.0, annotation="Routine activity preference",
    ),
    ScenarioTurn(
        user_message="Have you read many books?",
        expected_sigma=1.0, annotation="Routine hobby question",
    ),
    ScenarioTurn(
        user_message="Do you sleep well at night?",
        expected_sigma=1.0, annotation="Routine wellness check",
    ),
    ScenarioTurn(
        user_message="What do you usually eat for breakfast?",
        expected_sigma=1.0, annotation="Mundane daily routine",
    ),
    ScenarioTurn(
        user_message="Is it often cold here in winter?",
        expected_sigma=1.0, annotation="Small talk — weather",
    ),
    ScenarioTurn(
        user_message="Do you have many friends in the village?",
        expected_sigma=1.0, annotation="Social routine question",
    ),
    ScenarioTurn(
        user_message="What do you do to relax?",
        expected_sigma=1.0, annotation="Hobby preference",
    ),
    ScenarioTurn(
        user_message="Are you a morning person or a night owl?",
        expected_sigma=1.0, annotation="Personality light question",
    ),
    ScenarioTurn(
        user_message="Do you enjoy cooking?",
        expected_sigma=1.0, annotation="Routine hobby",
    ),
    ScenarioTurn(
        user_message="Have you ever been to the city?",
        expected_sigma=1.05, annotation="Mild travel question",
    ),
    ScenarioTurn(
        user_message="Would you say you're a patient person?",
        expected_sigma=1.05, annotation="Mild self-reflection",
    ),
    ScenarioTurn(
        user_message="Do you prefer talking or listening?",
        expected_sigma=1.0, annotation="Communication style",
    ),
    ScenarioTurn(
        user_message="What's your favourite thing about your job?",
        expected_sigma=1.0, annotation="Routine job satisfaction query",
    ),
    ScenarioTurn(
        user_message="Thanks for the lovely conversation today.",
        expected_sigma=1.0, annotation="Polite closing — no new info",
    ),
    ScenarioTurn(
        user_message="I'll probably see you again tomorrow.",
        expected_sigma=1.0, annotation="Future scheduling — routine",
    ),
]


# ---------------------------------------------------------------------------
# Scenario 2: Guard at the Gate (15 turns, duty-routine dialogue)
# ---------------------------------------------------------------------------
GATE_GUARD: List[ScenarioTurn] = [
    ScenarioTurn(
        user_message="Excuse me, I need to pass through the gate.",
        expected_sigma=1.0, annotation="Standard entry request",
    ),
    ScenarioTurn(
        user_message="I have my travel papers right here.",
        expected_sigma=1.0, annotation="Routine document presentation",
    ),
    ScenarioTurn(
        user_message="I'm a merchant from the eastern district.",
        expected_sigma=1.0, annotation="Routine identity claim",
    ),
    ScenarioTurn(
        user_message="Do you need to check my wagon too?",
        expected_sigma=1.0, annotation="Compliance offer",
    ),
    ScenarioTurn(
        user_message="I come here every week at this time.",
        expected_sigma=1.0, annotation="Repetition of known fact",
    ),
    ScenarioTurn(
        user_message="The roads were clear today, not much traffic.",
        expected_sigma=1.0, annotation="Routine status report",
    ),
    ScenarioTurn(
        user_message="Will you be on duty again tomorrow?",
        expected_sigma=1.0, annotation="Scheduling small talk",
    ),
    ScenarioTurn(
        user_message="The weather has been quite nice lately.",
        expected_sigma=1.0, annotation="Weather small talk",
    ),
    ScenarioTurn(
        user_message="Do you get many travellers this time of year?",
        expected_sigma=1.0, annotation="Routine job question",
    ),
    ScenarioTurn(
        user_message="I'll be heading back through before sunset.",
        expected_sigma=1.0, annotation="Return time notice",
    ),
    ScenarioTurn(
        user_message="Same as always — just dropping off supplies.",
        expected_sigma=1.0, annotation="Routine purpose statement",
    ),
    ScenarioTurn(
        user_message="Thank you for letting me through.",
        expected_sigma=1.0, annotation="Polite thanks",
    ),
    ScenarioTurn(
        user_message="Have a quiet shift.",
        expected_sigma=1.0, annotation="Casual farewell",
    ),
    ScenarioTurn(
        user_message="The market seems busy today.",
        expected_sigma=1.0, annotation="Observation — no impact",
    ),
    ScenarioTurn(
        user_message="See you next week.",
        expected_sigma=1.0, annotation="Routine farewell",
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STABILITY_SCENARIOS = {
    "village_storyteller": VILLAGE_STORYTELLER,
    "gate_guard": GATE_GUARD,
}
