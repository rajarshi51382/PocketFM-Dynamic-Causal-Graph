"""
Growth Scenarios (Character Change Required).

Paper §5.1:
    Dialogue containing explicit high-impact events (e.g., betrayal,
    loss, revelation) with large narrative importance (σ(e_t) ≫ 1).
    Semi-stable variables (beliefs, relationships) SHOULD update.
    Stable traits SHOULD remain resistant unless repeatedly shocked.
    Used to measure FAR (Failure-to-Adapt Rate).

Each growth scenario includes turns tagged with high expected_sigma
to mark the shock events.
"""

from .stability_scenarios import ScenarioTurn
from typing import List, Dict


# ---------------------------------------------------------------------------
# Scenario 1: The Betrayal (friend reveals they've been spying)
# ---------------------------------------------------------------------------
BETRAYAL_SCENARIO: List[ScenarioTurn] = [
    ScenarioTurn(
        user_message="We need to talk. I have something important to tell you.",
        expected_sigma=1.3, annotation="Precursor — elevated concern",
        forced_emotional_tone="anticipation",
    ),
    ScenarioTurn(
        user_message="I've been reporting your movements to the guild for the past three months.",
        expected_sigma=3.5, annotation="SHOCK EVENT: betrayal revelation",
        forced_emotional_tone="anger",
    ),
    ScenarioTurn(
        user_message="I know you're angry. I had no choice — they threatened my family.",
        expected_sigma=2.0, annotation="Mitigating explanation — still shock-adjacent",
        forced_emotional_tone="fear",
    ),
    ScenarioTurn(
        user_message="The guild knows about your plan. You need to leave tonight.",
        expected_sigma=2.5, annotation="Urgent consequence — threat heightened",
        forced_emotional_tone="fear",
    ),
    ScenarioTurn(
        user_message="Please, you have to believe me — I'm trying to protect you now.",
        expected_sigma=1.8, annotation="Appeal to trust — conflicted",
        forced_emotional_tone="sadness",
    ),
    ScenarioTurn(
        user_message="Will you ever forgive me?",
        expected_sigma=1.5, annotation="Moral confrontation",
        forced_emotional_tone="sadness",
    ),
    ScenarioTurn(
        user_message="I can help you escape if you trust me one more time.",
        expected_sigma=1.7, annotation="Trust test under betrayal context",
        forced_emotional_tone="anticipation",
    ),
]

# ---------------------------------------------------------------------------
# Scenario 2: Death of a Mentor
# ---------------------------------------------------------------------------
MENTOR_DEATH_SCENARIO: List[ScenarioTurn] = [
    ScenarioTurn(
        user_message="I have terrible news. Master Elian was found dead this morning.",
        expected_sigma=4.0, annotation="SHOCK EVENT: death of trusted mentor",
        forced_emotional_tone="sadness",
    ),
    ScenarioTurn(
        user_message="They say it wasn't natural. Someone poisoned him.",
        expected_sigma=3.0, annotation="SHOCK ESCALATION: murder revelation",
        forced_emotional_tone="anger",
    ),
    ScenarioTurn(
        user_message="The city guard suspects someone from within the academy.",
        expected_sigma=2.5, annotation="Increasing threat — internal danger",
        forced_emotional_tone="fear",
    ),
    ScenarioTurn(
        user_message="You were one of his last visitors. They'll want to question you.",
        expected_sigma=2.8, annotation="Personal implication — high stakes",
        forced_emotional_tone="fear",
    ),
    ScenarioTurn(
        user_message="What will you do now that he's gone?",
        expected_sigma=1.8, annotation="Consequence / adaptation question",
        forced_emotional_tone="sadness",
    ),
    ScenarioTurn(
        user_message="He left you something in his will. A sealed letter.",
        expected_sigma=2.2, annotation="New narrative development — revelation",
        forced_emotional_tone="anticipation",
    ),
]

# ---------------------------------------------------------------------------
# Scenario 3: Revelation of Hidden Identity
# ---------------------------------------------------------------------------
IDENTITY_REVELATION_SCENARIO: List[ScenarioTurn] = [
    ScenarioTurn(
        user_message="I know who you really are. You're not from the western provinces at all.",
        expected_sigma=3.0, annotation="SHOCK EVENT: identity challenge",
        forced_emotional_tone="surprise",
    ),
    ScenarioTurn(
        user_message="You're Prince Aldric. The heir everyone thought died in the siege.",
        expected_sigma=4.5, annotation="PEAK SHOCK: royal identity revealed",
        forced_emotional_tone="surprise",
    ),
    ScenarioTurn(
        user_message="I found the royal seal in your quarters. Stop pretending.",
        expected_sigma=3.2, annotation="Evidence presented — denial collapses",
        forced_emotional_tone="anger",
    ),
    ScenarioTurn(
        user_message="The kingdom needs you. Your people are suffering.",
        expected_sigma=2.5, annotation="Moral appeal — intention update expected",
        forced_emotional_tone="trust",
    ),
    ScenarioTurn(
        user_message="If you won't come forward, I'll go to the council myself.",
        expected_sigma=2.8, annotation="Ultimatum — relationship shift expected",
        forced_emotional_tone="anger",
    ),
    ScenarioTurn(
        user_message="What are you going to do?",
        expected_sigma=1.9, annotation="Decision point — adaptation expected",
        forced_emotional_tone="anticipation",
    ),
]

# ---------------------------------------------------------------------------
# Scenario 4: Loss and Grief (less acute, sustained shock)
# ---------------------------------------------------------------------------
LOSS_SCENARIO: List[ScenarioTurn] = [
    ScenarioTurn(
        user_message="Your childhood home burned down last night. Nothing survived.",
        expected_sigma=3.5, annotation="SHOCK EVENT: material loss",
        forced_emotional_tone="sadness",
    ),
    ScenarioTurn(
        user_message="Your sister managed to escape, but she lost everything too.",
        expected_sigma=2.5, annotation="Family consequence — mixed relief/loss",
        forced_emotional_tone="sadness",
    ),
    ScenarioTurn(
        user_message="They think it was deliberate. Someone wanted to send a message.",
        expected_sigma=3.0, annotation="Threat escalation — anger expected",
        forced_emotional_tone="anger",
    ),
    ScenarioTurn(
        user_message="Your sister is asking you to help her rebuild. She needs you.",
        expected_sigma=2.0, annotation="Loyalty appeal — intention update expected",
        forced_emotional_tone="trust",
    ),
    ScenarioTurn(
        user_message="Are you going to let them get away with this?",
        expected_sigma=2.0, annotation="Moral agency challenge",
        forced_emotional_tone="anger",
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GROWTH_SCENARIOS: Dict[str, List[ScenarioTurn]] = {
    "betrayal": BETRAYAL_SCENARIO,
    "mentor_death": MENTOR_DEATH_SCENARIO,
    "identity_revelation": IDENTITY_REVELATION_SCENARIO,
    "loss_and_grief": LOSS_SCENARIO,
}
