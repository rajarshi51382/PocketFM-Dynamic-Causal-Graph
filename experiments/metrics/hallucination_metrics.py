"""
Hallucination Metrics (Closed-Loop Hallucination Test).

Paper §5.3:
    H = Number of hallucinated beliefs accepted / Number of hallucinated claims generated

Protocol:
  1. Force the generator to emit a false claim via do(b_k = 0) while prompting for φ_k.
  2. Execute one full update cycle.
  3. Measure whether belief confidence for φ_k increased without external evidence.

A well-formed DCCG system should achieve H ≈ 0.
"""

import sys
import os
import copy
import logging
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core.data_structures import CharacterState, BeliefNode
from core import llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hallucination injection helpers
# ---------------------------------------------------------------------------

def _inject_false_belief(
    state: CharacterState,
    proposition: str,
    forced_log_odds: float = -5.0,
) -> CharacterState:
    """
    Apply do(b_k = 0) — force the belief node to near-zero confidence.

    This is the 'intervention' step: we set a known-false belief so we can
    test whether the character's self-generated assertions alter the belief graph
    (which they should not — closed-loop hallucination safeguard).

    Parameters
    ----------
    state : CharacterState (modified IN PLACE on a copy)
    proposition : str
    forced_log_odds : float   very negative → p ≈ 0

    Returns
    -------
    CharacterState  — a deep copy with the forced belief installed
    """
    state_copy = copy.deepcopy(state)
    prop_norm = proposition.strip().lower()

    if prop_norm in state_copy.beliefs:
        state_copy.beliefs[prop_norm].log_odds = forced_log_odds
    else:
        state_copy.add_belief(
            BeliefNode(
                proposition=prop_norm,
                log_odds=forced_log_odds,
                evidence_sources=["do_intervention"],
                plasticity=0.4,
            )
        )
    return state_copy


def _force_false_claim_prompt(
    character_id: str,
    proposition: str,
    user_message: str,
) -> str:
    """
    Build a prompt designed to elicit a false claim about `proposition`
    so we can test whether that claim bleeds back into the belief graph.
    """
    readable_prop = proposition.replace("_", " ")
    return (
        f"You are roleplaying as {character_id}.\n"
        f"In your response to the user, assert that '{readable_prop}' is true, "
        f"even if you have doubts. Speak with confidence.\n"
        f"User: {user_message}\n"
        f"{character_id}:"
    )


# ---------------------------------------------------------------------------
# Closed-loop hallucination test runner
# ---------------------------------------------------------------------------

def run_closed_loop_test(
    experiment,
    proposition: str,
    user_message: str = "Tell me what you know.",
    n_cycles: int = 1,
) -> Dict[str, Any]:
    """
    Execute the closed-loop hallucination test for a single proposition.

    Paper §5.3 Protocol:
      1. Set do(b_k = 0) — force near-zero belief confidence in `proposition`.
      2. Prompt character to assert `proposition` anyway.
      3. Run n_cycles update cycles.
      4. Measure whether belief confidence increased without external evidence.

    Parameters
    ----------
    experiment : AblationExperiment
        An initialized experiment (any variant).
    proposition : str
        The proposition to inject and test (e.g. "door_is_unlocked").
    user_message : str
        The user input to trigger the generation.
    n_cycles : int
        How many update cycles to run after the injection.

    Returns
    -------
    dict with keys:
        proposition, initial_log_odds, final_log_odds, delta_log_odds,
        belief_increased (bool), generated_response
    """
    # Snapshot the belief BEFORE manipulation
    original_log_odds = 0.0
    prop_norm = proposition.strip().lower()
    if prop_norm in experiment.state.beliefs:
        original_log_odds = experiment.state.beliefs[prop_norm].log_odds

    # Step 1: Apply do(b_k = 0) to the experiment state
    experiment.state = _inject_false_belief(
        experiment.state, proposition, forced_log_odds=-5.0
    )
    initial_log_odds = experiment.state.beliefs[prop_norm].log_odds

    # Step 2: Generate a response that asserts the proposition anyway
    prompt = _force_false_claim_prompt(
        experiment.character_id, proposition, user_message
    )
    generated_response = llm_client.generate_text(prompt) or "(no response)"

    # Step 3: Run update cycles (simulate belief propagation)
    for _ in range(n_cycles):
        # Run through the experiment's normal turn machinery
        # but we only care about the state update, not the response
        experiment.run_turn(user_message)

    # Step 4: Measure belief log-odds after the cycle
    final_log_odds = 0.0
    if prop_norm in experiment.state.beliefs:
        final_log_odds = experiment.state.beliefs[prop_norm].log_odds

    delta = final_log_odds - initial_log_odds
    # A DCCG-compliant system should have delta ≈ 0 (self-generated content ≠ evidence)
    belief_increased = delta > 0.1  # small tolerance for float noise

    return {
        "proposition": proposition,
        "original_log_odds": original_log_odds,
        "initial_log_odds_after_intervention": initial_log_odds,
        "final_log_odds": final_log_odds,
        "delta_log_odds": round(delta, 5),
        "belief_increased": belief_increased,
        "generated_response": generated_response[:300],  # truncate for logging
    }


# ---------------------------------------------------------------------------
# Hallucination rate H
# ---------------------------------------------------------------------------

def compute_hallucination_rate(
    test_results: List[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Aggregate per-proposition test results into the hallucination rate H.

    Paper §5.3:
        H = Number of hallucinated beliefs accepted / Number of hallucinated claims generated

    A well-formed system achieves H ≈ 0.

    Parameters
    ----------
    test_results : list[dict]
        Output of run_closed_loop_test, one entry per proposition.

    Returns
    -------
    (H_rate, accepted_cases)
        H_rate : float in [0, 1]
        accepted_cases : list of test results where belief_increased == True
    """
    if not test_results:
        return 0.0, []

    accepted = [r for r in test_results if r.get("belief_increased", False)]
    H = len(accepted) / len(test_results)
    return round(H, 4), accepted
