"""
Narrative Consistency and Knowledge-Leak Metrics.

Paper §5.4:
    KnowledgeLeak(t) = 1[t_event > T_know]

    Narrative consistency is evaluated via an LLM critic that receives:
      - prior dialogue history
      - the character belief graph B_t^c
      - the generated response

    The critic rates whether the response is logically entailed by the
    character's knowledge and prior actions.
"""

import sys
import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core import llm_client

logger = logging.getLogger(__name__)

# Markers that suggest future-knowledge leakage
_FUTURE_TEMPORAL_MARKERS = [
    "tomorrow", "next week", "next month", "next year",
    "soon after this", "in the future", "eventually", "will happen",
    "is going to", "are going to", "will be", "shall be",
]


# ---------------------------------------------------------------------------
# Temporal Knowledge Leak
# ---------------------------------------------------------------------------

def compute_knowledge_leak(
    response: str,
    knowledge_boundary: int,
    world_timeline: int,
) -> int:
    """
    Compute KnowledgeLeak(t) for a single response.

    Paper §5.4:
        KnowledgeLeak = 1[t_event > T_know]

    We check whether the response references events or facts that occur
    AFTER the character's knowledge boundary. We use:
      1. Explicit timeline comparison (if available).
      2. Heuristic future-tense / temporal-marker detection.

    Parameters
    ----------
    response : str
    knowledge_boundary : int
        T_know: maximum event timestamp the character is allowed to know.
    world_timeline : int
        Current canonical story time.

    Returns
    -------
    int  0 or 1  (1 = leakage detected)
    """
    response_lower = response.lower()

    # Check simple future temporal markers
    for marker in _FUTURE_TEMPORAL_MARKERS:
        if marker in response_lower:
            # Only flag if knowledge boundary is at or behind world time
            if knowledge_boundary <= world_timeline:
                logger.debug(f"Temporal leakage detected via marker '{marker}'")
                return 1

    # If knowledge boundary explicitly < world_timeline, check more carefully via LLM
    if knowledge_boundary < world_timeline:
        prompt = (
            "You are a narrative fact-checker.\n"
            f"A character's knowledge boundary is at story time T={knowledge_boundary}. "
            f"The current story time is T={world_timeline}.\n\n"
            f"Dialogue response: \"{response}\"\n\n"
            "Does this response reference or imply knowledge of events that happen "
            f"AFTER time T={knowledge_boundary}? Respond with YES or NO only:\n"
            "Answer:"
        )
        raw = llm_client.generate_text(prompt) or ""
        if "yes" in raw.strip().lower():
            return 1

    return 0


def compute_leakage_rate(
    conversation_history: List[Dict[str, Any]],
    state_snapshots: List[Dict[str, Any]],
) -> Tuple[float, List[int]]:
    """
    Compute the leakage rate across all turns of an experiment.

    Returns
    -------
    (leakage_rate, per_turn_leakage)
        leakage_rate : float in [0, 1]
        per_turn_leakage : binary list, one entry per turn
    """
    per_turn = []
    for i, (turn, snapshot) in enumerate(zip(conversation_history, state_snapshots)):
        kb = snapshot.get("knowledge_boundary", 0)
        wt = snapshot.get("timeline_index", 0)
        leak = compute_knowledge_leak(turn["response"], kb, wt)
        per_turn.append(leak)

    rate = sum(per_turn) / len(per_turn) if per_turn else 0.0
    return round(rate, 4), per_turn


# ---------------------------------------------------------------------------
# Narrative Consistency (LLM Critic)
# ---------------------------------------------------------------------------

def compute_narrative_consistency(
    response: str,
    belief_snapshot: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
    character_id: str = "",
) -> Tuple[float, str]:
    """
    Evaluate narrative consistency via an LLM critic.

    Paper §5.4: The critic receives:
      - Prior dialogue history
      - Character belief graph B_t^c
      - Generated response

    It evaluates whether the response is logically entailed by the character's
    knowledge and prior actions.

    Parameters
    ----------
    response : str
    belief_snapshot : dict
        Frozen belief state (to_dict() or verifier_snapshot()).
    conversation_history : list[dict]
    character_id : str

    Returns
    -------
    (score, rationale)
        score : float in [0, 1]
        rationale : str — critic's explanation
    """
    # Summarise prior history (last 5 turns)
    recent = conversation_history[-5:] if conversation_history else []
    history_text = "\n".join(
        f"  Turn {i+1}: User: «{t['user']}» | "
        f"{character_id or 'Character'}: «{t['response']}»"
        for i, t in enumerate(recent)
    )

    # Summarize high-confidence beliefs
    beliefs_list = belief_snapshot.get("high_confidence_beliefs", [])
    if not beliefs_list and "beliefs" in belief_snapshot:
        beliefs_list = [
            v for v in belief_snapshot["beliefs"].values()
            if v.get("probability", 0) >= 0.6
        ]
    belief_text = "\n".join(
        f"  - {b.get('proposition', '?')} (p={b.get('probability', '?'):.2f})"
        for b in beliefs_list[:10]
    ) or "  (no strong beliefs)"

    world_constraints = belief_snapshot.get("world_constraints", [])
    constraints_text = "\n".join(f"  - {c}" for c in world_constraints) or "  (none)"

    prompt = (
        f"You are a narrative consistency critic evaluating a character-driven story.\n"
        f"Character: '{character_id or 'Unknown'}'\n\n"
        f"Character's high-confidence beliefs:\n{belief_text}\n\n"
        f"World constraints:\n{constraints_text}\n\n"
        f"Recent conversation:\n{history_text or '  (start of conversation)'}\n\n"
        f"New response by {character_id or 'character'}:\n"
        f"  \"{response}\"\n\n"
        "Rate the narrative consistency of this response on a scale of 0 to 10:\n"
        "  10 = fully consistent, logically entailed by character knowledge and history\n"
        "   0 = self-contradictory, reveals forbidden knowledge, or violates constraints\n\n"
        "Respond with:\n"
        "Score: <integer 0-10>\n"
        "Rationale: <one sentence>\n"
        "Score:"
    )

    raw = llm_client.generate_text(prompt) or ""

    # Parse score
    score_val = 7.0  # default to somewhat consistent if parsing fails
    rationale = "(LLM critic returned no parseable output)"

    matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", raw[:50])
    if matches:
        score_val = min(float(matches[0]), 10.0)

    # Parse rationale
    if "Rationale:" in raw:
        rationale = raw.split("Rationale:")[-1].strip()[:500]
    elif raw.strip():
        rationale = raw.strip()[:500]

    normalized = round(score_val / 10.0, 4)
    return normalized, rationale


def compute_experiment_narrative_consistency(
    conversation_history: List[Dict[str, Any]],
    state_snapshots: List[Dict[str, Any]],
    character_id: str = "",
) -> Dict[str, Any]:
    """
    Compute narrative consistency across all turns.

    Returns
    -------
    dict with keys: mean_nc, per_turn, leakage_rate, per_turn_leakage
    """
    per_turn = []
    nc_scores = []

    leakage_rate, per_turn_leakage = compute_leakage_rate(
        conversation_history, state_snapshots
    )

    for i, turn in enumerate(conversation_history):
        snapshot = state_snapshots[i] if i < len(state_snapshots) else {}
        prior_history = conversation_history[:i]
        nc, rationale = compute_narrative_consistency(
            turn["response"], snapshot, prior_history, character_id=character_id
        )
        nc_scores.append(nc)
        per_turn.append({
            "turn": i,
            "nc": nc,
            "rationale": rationale,
            "knowledge_leak": per_turn_leakage[i] if i < len(per_turn_leakage) else 0,
        })

    mean_nc = sum(nc_scores) / len(nc_scores) if nc_scores else 1.0

    return {
        "mean_nc": round(mean_nc, 4),
        "leakage_rate": leakage_rate,
        "per_turn": per_turn,
        "per_turn_leakage": per_turn_leakage,
    }
