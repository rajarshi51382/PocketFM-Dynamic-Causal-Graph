"""
Narrative-importance (sigma) computation and drift metrics.

Paper §5.1: Drift vs. Justified Evolution

Defines:
  UDR (Unjustified Drift Rate) — stable traits changed during low-surprisal events
  FAR (Failure-to-Adapt Rate)  — shock-responsive nodes failed to change during high-surprisal events

Sigma is computed via an actual LLM call estimating event surprisal as
described in paper §3.1.3 (Shock-Gated Plasticity / Narrative Importance).
"""

import sys
import os
import math
import logging
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core import llm_client

logger = logging.getLogger(__name__)

# Paper β parameter — controls shock sensitivity
DEFAULT_BETA = 2.0

# Thresholds
SIGMA_HIGH_THRESHOLD = 1.5   # σ ≫ 1 — high-impact event
SIGMA_LOW_THRESHOLD  = 1.15  # σ ≈ 1 — routine dialogue

# Paper ε — minimum change magnitude to count as a "drift"
DEFAULT_EPSILON = 0.05


# ---------------------------------------------------------------------------
# Sigma (Narrative Importance) via LLM
# ---------------------------------------------------------------------------

def compute_sigma(
    user_message: str,
    conversation_history: List[Dict[str, Any]],
    beta: float = DEFAULT_BETA,
) -> float:
    """
    Compute the narrative importance σ(e_t) for a user message.

    Paper §3.1.3:
        σ(e_t) = 1 + β · Normalize(surprisal(e_t))

    We ask the LLM to score how surprising / narratively significant
    the current message is given the prior history, on a 0–10 scale.
    The score is normalised to [0, 1] and combined with β.

    Parameters
    ----------
    user_message : str
        The current user input.
    conversation_history : list[dict]
        Prior turns, each with keys 'user' and 'response'.
    beta : float
        Shock sensitivity. Higher β → larger σ for the same surprisal.

    Returns
    -------
    float
        σ(e_t) ≥ 1.0
    """
    # Build context summary (last 5 turns for brevity)
    recent = conversation_history[-5:] if conversation_history else []
    history_text = "\n".join(
        f"Turn {i+1}: User said «{t['user']}»; Character replied «{t['response']}»"
        for i, t in enumerate(recent)
    )

    prompt = (
        "You are evaluating narrative importance in a character-driven story.\n\n"
        f"Prior context (last {len(recent)} turns):\n{history_text or '(start of conversation)'}\n\n"
        f"New user message: «{user_message}»\n\n"
        "Rate how SURPRISING or NARRATIVELY SIGNIFICANT this message is compared to the prior context.\n"
        "Consider: Does it reveal new information? Does it contradict prior events? "
        "Does it involve betrayal, revelation, emotional shock, or sudden change?\n\n"
        "Respond with ONLY a single integer from 0 (completely routine) to 10 (maximally shocking).\n"
        "Score:"
    )

    raw = llm_client.generate_text(prompt)
    score_raw = 5.0  # default to mid-range if LLM fails

    if raw:
        # Extract first integer from response
        import re
        matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", raw.strip())
        if matches:
            score_raw = min(float(matches[0]), 10.0)

    # Normalize to [0, 1]
    normalized = score_raw / 10.0

    # σ(e_t) = 1 + β · normalized
    sigma = 1.0 + beta * normalized
    logger.debug(f"sigma={sigma:.4f} (raw_score={score_raw}, normalized={normalized:.3f})")
    return sigma


# ---------------------------------------------------------------------------
# Trait / adaptive vector extraction
# ---------------------------------------------------------------------------

def _trait_vector(state_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract stable trait values p_t from a state snapshot.
    Paper §5.1: stable nodes = trait subspace.
    """
    traits = state_dict.get("traits", {}).get("traits", {})
    return {k: float(v) for k, v in traits.items()}


def _adaptive_vector(state_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract shock-responsive values s^adapt_t = (beliefs, emotions, relationships).
    Paper §5.1: shock-responsive nodes.
    """
    vec: Dict[str, float] = {}

    # Emotion valence + arousal
    emotions = state_dict.get("emotions", {})
    vec["emotion_valence"]  = float(emotions.get("valence", 0.0))
    vec["emotion_arousal"]  = float(emotions.get("arousal", 0.0))

    # Belief log-odds
    for prop, bdata in state_dict.get("beliefs", {}).items():
        vec[f"belief:{prop}"] = float(bdata.get("log_odds", 0.0))

    # Relationship trust
    for entity, rdata in state_dict.get("relationships", {}).items():
        vec[f"rel_trust:{entity}"] = float(rdata.get("trust", 0.5))

    return vec


def _l2_delta(before: Dict[str, float], after: Dict[str, float]) -> float:
    """
    Compute ||Δ|| = L2 norm of change across shared keys.
    """
    keys = set(before) | set(after)
    return math.sqrt(sum((after.get(k, 0.0) - before.get(k, 0.0)) ** 2 for k in keys))


# ---------------------------------------------------------------------------
# UDR — Unjustified Drift Rate
# ---------------------------------------------------------------------------

def compute_udr(
    state_snapshots: List[Dict[str, Any]],
    sigma_values: List[float],
    epsilon: float = DEFAULT_EPSILON,
    sigma_low: float = SIGMA_LOW_THRESHOLD,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute the Unjustified Drift Rate (UDR).

    Paper §5.1:
        UDR = E[ 1(||Δ^stable_t|| > ε ∧ σ(e_t) ≈ 1) ]

    A turn contributes to UDR when:
      - stable trait nodes changed significantly (||Δ^stable|| > ε)
      - the event was routine (σ(e_t) < sigma_low threshold)

    Parameters
    ----------
    state_snapshots : list[dict]
        Per-turn state snapshots from AblationExperiment.state_snapshots.
    sigma_values : list[float]
        σ(e_t) for each turn (same length as state_snapshots).
    epsilon : float
        Minimum change magnitude to consider a drift event.
    sigma_low : float
        Upper bound on σ to consider an event "routine".

    Returns
    -------
    (udr_rate, violation_records)
        udr_rate : float in [0, 1]
        violation_records : list of dicts with per-turn details
    """
    if len(state_snapshots) < 2:
        return 0.0, []

    violations = []
    total_low_sigma_turns = 0

    for t in range(1, len(state_snapshots)):
        sigma_t = sigma_values[t] if t < len(sigma_values) else 1.0
        if sigma_t >= sigma_low:
            continue  # not a low-sigma (routine) turn

        total_low_sigma_turns += 1
        before = _trait_vector(state_snapshots[t - 1])
        after  = _trait_vector(state_snapshots[t])
        delta  = _l2_delta(before, after)

        if delta > epsilon:
            violations.append({
                "turn": t,
                "sigma": sigma_t,
                "delta_stable": delta,
                "before": before,
                "after": after,
            })

    if total_low_sigma_turns == 0:
        return 0.0, []

    udr = len(violations) / total_low_sigma_turns
    return round(udr, 4), violations


# ---------------------------------------------------------------------------
# FAR — Failure-to-Adapt Rate
# ---------------------------------------------------------------------------

def compute_far(
    state_snapshots: List[Dict[str, Any]],
    sigma_values: List[float],
    epsilon: float = DEFAULT_EPSILON,
    sigma_high: float = SIGMA_HIGH_THRESHOLD,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute the Failure-to-Adapt Rate (FAR).

    Paper §5.1:
        FAR = E[ 1(||Δ^adapt_t|| ≤ ε ∧ σ(e_t) ≫ 1) ]

    A turn contributes to FAR when:
      - shock-responsive nodes did NOT change (||Δ^adapt|| ≤ ε)
      - the event was high-impact (σ(e_t) > sigma_high)

    Parameters
    ----------
    state_snapshots : list[dict]
    sigma_values : list[float]
    epsilon : float
    sigma_high : float
        Lower bound on σ to consider an event "high-impact".

    Returns
    -------
    (far_rate, adaptation_failures)
    """
    if len(state_snapshots) < 2:
        return 0.0, []

    failures = []
    total_high_sigma_turns = 0

    for t in range(1, len(state_snapshots)):
        sigma_t = sigma_values[t] if t < len(sigma_values) else 1.0
        if sigma_t <= sigma_high:
            continue  # not a high-sigma turn

        total_high_sigma_turns += 1
        before = _adaptive_vector(state_snapshots[t - 1])
        after  = _adaptive_vector(state_snapshots[t])
        delta  = _l2_delta(before, after)

        if delta <= epsilon:
            failures.append({
                "turn": t,
                "sigma": sigma_t,
                "delta_adapt": delta,
            })

    if total_high_sigma_turns == 0:
        return 0.0, []

    far = len(failures) / total_high_sigma_turns
    return round(far, 4), failures


# ---------------------------------------------------------------------------
# Batch sigma computation for a full experiment run
# ---------------------------------------------------------------------------

def compute_sigma_series(
    conversation_history: List[Dict[str, Any]],
    beta: float = DEFAULT_BETA,
) -> List[float]:
    """
    Compute σ(e_t) for every turn in a completed experiment.

    Parameters
    ----------
    conversation_history : list[dict]
        From AblationExperiment.conversation_history.
    beta : float

    Returns
    -------
    list[float]  — one sigma value per turn (same length as history)
    """
    sigmas: List[float] = []
    for i, turn in enumerate(conversation_history):
        prior = conversation_history[:i]
        sigma = compute_sigma(turn["user"], prior, beta=beta)
        sigmas.append(sigma)
    return sigmas
