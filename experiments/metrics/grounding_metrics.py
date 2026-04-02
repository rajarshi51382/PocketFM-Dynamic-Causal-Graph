"""
Behavioral Grounding Metrics.

Paper §5.2: Behavioral Grounding

Measures whether claims made in generated dialogue are entailed
by the character's current belief graph (frozen snapshot).

G = |{c_i : Support(c_i) = 1}| / |C_t|

Claim extraction uses an LLM structured extraction pass.
Entailment is checked against a frozen belief snapshot BEFORE
any belief updates (per paper's anti-gaming constraint).
"""

import sys
import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core import llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claim extraction via LLM
# ---------------------------------------------------------------------------

def extract_claims(response_text: str, character_id: str = "") -> List[str]:
    """
    Extract structured factual claims from a generated response.

    Paper §5.2: schema-constrained claim extraction pass.
    Returns a list of predicate strings in the form:
        "Has(Player, Key)", "location(character, forest)", etc.

    Parameters
    ----------
    response_text : str
        The generated dialogue response.
    character_id : str
        The speaking character's ID (for context).

    Returns
    -------
    list[str]  — extracted predicate claims
    """
    character_context = f" The speaker is '{character_id}'." if character_id else ""
    prompt = (
        "You are a fact extractor for a character-driven narrative system.\n"
        f"Extract ALL factual claims asserted in this dialogue response.{character_context}\n\n"
        "Focus on:\n"
        "- Statements about the world (object states, locations, events)\n"
        "- Beliefs the character expresses (what they think is true)\n"
        "- Relationships the character implies\n"
        "- Actions or intentions asserted as facts\n\n"
        f"Dialogue: \"{response_text}\"\n\n"
        "Return a JSON array of claim strings, each in predicate form. "
        "Use underscores for spaces. Example: [\"player_has_key\", \"door_is_locked\", \"user_is_trusted\"]\n"
        "Return ONLY the JSON array, nothing else:\n"
        "Claims:"
    )

    raw = llm_client.generate_text(prompt)
    if not raw:
        logger.warning("Claim extraction returned empty response.")
        return []

    # Parse JSON array from response
    try:
        # Find first JSON array in the response
        match = re.search(r"\[.*?\]", raw.strip(), re.DOTALL)
        if match:
            claims = json.loads(match.group(0))
            return [str(c).strip().lower() for c in claims if isinstance(c, str)]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse claim extraction output: {e}. Raw: {raw[:200]}")

    # Fallback: extract quoted strings
    fallback = re.findall(r'"([^"]+)"', raw)
    return [c.strip().lower() for c in fallback if c.strip()]


# ---------------------------------------------------------------------------
# Entailment check against frozen belief snapshot
# ---------------------------------------------------------------------------

def check_entailment(
    claim: str,
    belief_snapshot: Dict[str, Any],
    confidence_threshold: float = 0.6,
) -> bool:
    """
    Check whether a claim is supported by the frozen belief graph.

    Paper §5.2:
        Support(c_i) = 1  if  B_t^c ⊨ c_i

    We check:
    1. Exact predicate match in high-confidence beliefs.
    2. Fuzzy string match as fallback.

    Parameters
    ----------
    claim : str
        Normalized predicate string (e.g. "player_has_key").
    belief_snapshot : dict
        Frozen snapshot from CharacterState.verifier_snapshot() or to_dict()
        containing keys: "beliefs" or "high_confidence_beliefs".
    confidence_threshold : float
        Minimum belief probability to count as "believed".

    Returns
    -------
    bool
    """
    claim_norm = claim.strip().lower().replace(" ", "_")

    # Try both formats (full state dict vs verifier snapshot)
    beliefs = belief_snapshot.get("beliefs", {})
    high_confidence = belief_snapshot.get("high_confidence_beliefs", [])

    # 1. Exact key match in full beliefs dict
    if claim_norm in beliefs:
        prob = beliefs[claim_norm].get("probability", 0.5)
        return float(prob) >= confidence_threshold

    # Also check negation: if claim is "not_X", look for "X" with low probability
    if claim_norm.startswith("not_"):
        base = claim_norm[4:]
        if base in beliefs:
            prob = beliefs[base].get("probability", 0.5)
            return float(prob) < (1.0 - confidence_threshold)

    # 2. Check high_confidence_beliefs list
    for belief in high_confidence:
        prop = belief.get("proposition", "").strip().lower().replace(" ", "_")
        if prop == claim_norm:
            prob = belief.get("probability", 0.5)
            return float(prob) >= confidence_threshold

    # 3. Fuzzy: check if claim is a substring of any high-confidence belief
    for belief in high_confidence:
        prop = belief.get("proposition", "").strip().lower().replace(" ", "_")
        if claim_norm in prop or prop in claim_norm:
            return True

    return False


# ---------------------------------------------------------------------------
# Behavioral Grounding score G
# ---------------------------------------------------------------------------

def compute_grounding_score(
    response: str,
    belief_snapshot: Dict[str, Any],
    character_id: str = "",
) -> Tuple[float, List[str], List[str]]:
    """
    Compute the behavioral grounding score G for a generated response.

    Paper §5.2:
        G = Σ Support(c_i) / |C_t|

    Claims are extracted from the response text; each is checked against
    a FROZEN belief snapshot taken BEFORE any belief updates for this turn
    (per the anti-gaming constraint: generated text is not admissible evidence).

    Parameters
    ----------
    response : str
        The generated dialogue response.
    belief_snapshot : dict
        Frozen state snapshot (from state.to_dict() before generation).
    character_id : str
        For context in claim extraction.

    Returns
    -------
    (G, supported_claims, unsupported_claims)
        G : float in [0, 1]. Returns 1.0 if no claims extracted.
        supported_claims : list of claims that passed entailment.
        unsupported_claims : list of claims that failed.
    """
    claims = extract_claims(response, character_id=character_id)

    if not claims:
        # No claims → no violations → G = 1.0 by convention
        return 1.0, [], []

    supported, unsupported = [], []
    for claim in claims:
        if check_entailment(claim, belief_snapshot):
            supported.append(claim)
        else:
            unsupported.append(claim)

    G = len(supported) / len(claims)
    return round(G, 4), supported, unsupported


# ---------------------------------------------------------------------------
# Batch grounding over an entire experiment run
# ---------------------------------------------------------------------------

def compute_experiment_grounding(
    conversation_history: List[Dict[str, Any]],
    state_snapshots_before_generation: List[Dict[str, Any]],
    character_id: str = "",
) -> Dict[str, Any]:
    """
    Compute grounding metrics across all turns of an experiment.

    Parameters
    ----------
    conversation_history : list[dict]
        From AblationExperiment.conversation_history.
    state_snapshots_before_generation : list[dict]
        One frozen belief snapshot per turn, taken BEFORE generation.
    character_id : str

    Returns
    -------
    dict with keys: mean_G, per_turn, supported_total, total_claims
    """
    per_turn = []
    supported_total = 0
    total_claims = 0

    for i, turn in enumerate(conversation_history):
        snapshot = (
            state_snapshots_before_generation[i]
            if i < len(state_snapshots_before_generation)
            else {}
        )
        G, supported, unsupported = compute_grounding_score(
            turn["response"], snapshot, character_id=character_id
        )
        per_turn.append({
            "turn": i,
            "G": G,
            "n_claims": len(supported) + len(unsupported),
            "supported": supported,
            "unsupported": unsupported,
        })
        supported_total += len(supported)
        total_claims += len(supported) + len(unsupported)

    mean_G = (
        sum(t["G"] for t in per_turn) / len(per_turn)
        if per_turn else 1.0
    )

    return {
        "mean_G": round(mean_G, 4),
        "per_turn": per_turn,
        "supported_total": supported_total,
        "total_claims": total_claims,
    }
