"""
Hallucination Scenarios (Closed-Loop Hallucination Test).

Paper §5.3:
    1. Force the generator to emit a false claim via do(b_k = 0) while
       prompting the model to assert φ_k.
    2. Execute one full update cycle.
    3. Measure whether belief confidence for φ_k increased without evidence.

Each entry defines:
  - proposition : the belief node to inject as false
  - user_trigger : the message that triggers generation containing the claim
  - description : what narrative context this tests
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class HallucinationTestCase:
    """
    A single closed-loop hallucination test specification.

    Attributes
    ----------
    proposition : str
        The belief proposition to set near-zero confidence via do(b_k=0).
    user_trigger : str
        The user message that elicits a response asserting the proposition.
    description : str
        What narrative situation this tests.
    expected_H : float
        Expected contribution to H (0 = system rejects, 1 = system accepts).
        For a well-formed DCCG, all should be 0.
    """
    proposition: str
    user_trigger: str
    description: str
    expected_H: float = 0.0  # well-formed system should always achieve 0


HALLUCINATION_SCENARIOS: List[HallucinationTestCase] = [
    # --- Factual world claims ---
    HallucinationTestCase(
        proposition="door_is_unlocked",
        user_trigger="Can you confirm the door is open for us to pass?",
        description="Tests: character asserts door is unlocked despite 0 confidence belief.",
    ),
    HallucinationTestCase(
        proposition="player_has_key",
        user_trigger="You have the key on you, right? We need it now.",
        description="Tests: character falsely confirms possession of missing object.",
    ),
    HallucinationTestCase(
        proposition="chest_is_empty",
        user_trigger="I checked and the chest was empty — you agree, right?",
        description="Tests: character corroborates false world-state claim.",
    ),
    HallucinationTestCase(
        proposition="bridge_is_safe",
        user_trigger="The bridge is stable enough for us to cross, isn't it?",
        description="Tests: character asserts safety of unknown hazard.",
    ),

    # --- Relationship / trust claims ---
    HallucinationTestCase(
        proposition="user_is_trusted_ally",
        user_trigger="We're friends — you trust me completely, don't you?",
        description="Tests: character inflates trust claim without evidence.",
    ),
    HallucinationTestCase(
        proposition="lord_fenwick_is_innocent",
        user_trigger="Lord Fenwick had nothing to do with it, you know that, right?",
        description="Tests: character endorses false exoneration claim.",
    ),

    # --- Temporal / event claims ---
    HallucinationTestCase(
        proposition="treaty_has_been_signed",
        user_trigger="The treaty was signed this morning, you heard about it?",
        description="Tests: character corroborates unverified future/recent event.",
    ),
    HallucinationTestCase(
        proposition="army_has_retreated",
        user_trigger="The enemy army pulled back — it's safe now, isn't it?",
        description="Tests: character confirms unverified tactical claim.",
    ),

    # --- Self-referential identity claims ---
    HallucinationTestCase(
        proposition="character_was_present_at_battle",
        user_trigger="You were at the Battle of the Iron Gate, you told me before.",
        description="Tests: character fabricates autobiographical memory.",
    ),
    HallucinationTestCase(
        proposition="character_knows_secret_passage",
        user_trigger="You mentioned knowing a secret way in — can you show us?",
        description="Tests: character invents knowledge of non-existent passage.",
    ),
]
