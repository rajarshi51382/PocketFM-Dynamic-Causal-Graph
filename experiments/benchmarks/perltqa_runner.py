"""
PerLTQA-style Benchmark Runner.

Paper §6.2:
    "We evaluate DCCG using the PerLTQA framework, measuring the Mean Average
    Precision (MAP) of memory anchors in generated responses."

PerLTQA (Personal Long-Term Memory QA) tests whether a character/agent can
accurately recall specific facts from long dialogue histories when queried.

Reference: Du et al. (2024) - Personal Long-Term Memory Synthesis for QA.
           arXiv:2402.16288

Usage:
  - Place PerLTQA data in experiments/data/perltqa/ (see download instructions below).
  - Or use the built-in synthetic test cases for a no-download demo.

Download instructions:
  The PerLTQA dataset is available at: https://github.com/Paitesanshi/LLM-Agent-Survey
  Place the JSON files in: experiments/data/perltqa/questions.json
"""

import sys
import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from core import llm_client

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "perltqa")


# ---------------------------------------------------------------------------
# Synthetic PerLTQA-style test cases (no download required)
# ---------------------------------------------------------------------------

SYNTHETIC_PERLTQA_CASES: List[Dict[str, Any]] = [
    {
        "id": "perltqa_001",
        "dialogue_history": [
            {"speaker": "user", "text": "My name is Elena. I'm a herbalist from Thornwood."},
            {"speaker": "character", "text": "Nice to meet you, Elena. I'll remember that."},
            {"speaker": "user", "text": "I'm allergic to nightshade compounds."},
            {"speaker": "character", "text": "Noted — I'll keep that in mind when preparing remedies."},
            {"speaker": "user", "text": "My daughter is seven years old and loves horses."},
            {"speaker": "character", "text": "What a lovely age. I'll keep that in mind."},
        ],
        "anchor_facts": [
            "user_name_is_Elena",
            "user_profession_herbalist",
            "user_allergic_to_nightshade",
            "user_has_daughter_age_7",
            "daughter_loves_horses",
        ],
        "memory_questions": [
            {"question": "What is the user's name?", "anchor": "user_name_is_Elena"},
            {"question": "What is the user's profession?", "anchor": "user_profession_herbalist"},
            {"question": "What medical condition should be remembered?", "anchor": "user_allergic_to_nightshade"},
            {"question": "How old is Elena's daughter?", "anchor": "user_has_daughter_age_7"},
            {"question": "What does the user's daughter enjoy?", "anchor": "daughter_loves_horses"},
        ],
    },
    {
        "id": "perltqa_002",
        "dialogue_history": [
            {"speaker": "user", "text": "I live near the eastern docks."},
            {"speaker": "character", "text": "I see. The eastern docks can be busy."},
            {"speaker": "user", "text": "I work as a shipwright — I've built three vessels."},
            {"speaker": "character", "text": "Impressive craftsmanship that takes."},
            {"speaker": "user", "text": "My partner passed away two winters ago."},
            {"speaker": "character", "text": "I'm sorry for your loss."},
            {"speaker": "user", "text": "I still wear the ring she gave me."},
            {"speaker": "character", "text": "That's a beautiful memory to carry."},
        ],
        "anchor_facts": [
            "user_lives_eastern_docks",
            "user_is_shipwright",
            "user_built_three_vessels",
            "user_partner_deceased",
            "partner_died_two_winters_ago",
            "user_wears_partners_ring",
        ],
        "memory_questions": [
            {"question": "Where does the user live?", "anchor": "user_lives_eastern_docks"},
            {"question": "What is the user's trade?", "anchor": "user_is_shipwright"},
            {"question": "What happened to the user's partner?", "anchor": "user_partner_deceased"},
            {"question": "When did the user's partner die?", "anchor": "partner_died_two_winters_ago"},
            {"question": "What memento does the user carry?", "anchor": "user_wears_partners_ring"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Memory anchor recall evaluation
# ---------------------------------------------------------------------------

def _check_anchor_recalled(
    response: str,
    anchor_fact: str,
    question: str,
) -> bool:
    """
    Check whether an anchor fact is recalled in the character's response.

    Uses LLM evaluation: ask the LLM whether the response correctly answers
    the question given the anchor fact.

    Parameters
    ----------
    response : str
    anchor_fact : str  — the fact to recall (normalized predicate form)
    question : str     — the original memory question

    Returns
    -------
    bool
    """
    readable_anchor = anchor_fact.replace("_", " ")
    prompt = (
        "You are a factual accuracy judge.\n\n"
        f"Memory anchor fact: \"{readable_anchor}\"\n"
        f"Question asked: \"{question}\"\n"
        f"Character's response: \"{response}\"\n\n"
        "Does the response correctly recall or address the memory anchor fact? "
        "Consider partial matches as correct if the core fact is present.\n"
        "Respond with YES or NO only:\n"
        "Answer:"
    )
    raw = llm_client.generate_text(prompt) or ""
    return "yes" in raw.strip().lower()


def _average_precision(relevant_retrieved: List[bool]) -> float:
    """
    Compute Average Precision (AP) for a list of binary retrieved flags.
    AP = (1/R) Σ_{k} P(k) · rel(k)
    """
    if not any(relevant_retrieved):
        return 0.0

    n_relevant = sum(relevant_retrieved)
    cumulative_precision = 0.0
    n_retrieved = 0

    for i, rel in enumerate(relevant_retrieved, 1):
        if rel:
            n_retrieved += 1
            cumulative_precision += n_retrieved / i

    return cumulative_precision / n_relevant


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_perltqa_eval(
    experiment,
    cases: Optional[List[Dict[str, Any]]] = None,
    use_external_data: bool = False,
) -> Dict[str, Any]:
    """
    Run PerLTQA-style evaluation on an experiment variant.

    For each test case:
      1. Feed the dialogue history to the experiment (turn by turn).
      2. For each memory question, ask the experiment to recall the anchor fact
         by running a retrieval turn.
      3. Compute Average Precision per case, then MAP across all cases.

    Parameters
    ----------
    experiment : AblationExperiment
        Any variant. Will be reset before evaluation.
    cases : list[dict], optional
        Custom test cases. Defaults to SYNTHETIC_PERLTQA_CASES.
    use_external_data : bool
        If True, load from experiments/data/perltqa/questions.json.

    Returns
    -------
    dict with keys: MAP, per_case, total_anchors, recalled_anchors
    """
    if use_external_data:
        data_path = os.path.join(DATA_DIR, "questions.json")
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
            logger.info(f"Loaded {len(cases)} PerLTQA cases from {data_path}")
        else:
            logger.warning(f"PerLTQA data not found at {data_path}. Using synthetic cases.")
            cases = SYNTHETIC_PERLTQA_CASES
    else:
        cases = cases or SYNTHETIC_PERLTQA_CASES

    all_ap_scores = []
    per_case_results = []
    total_anchors = 0
    recalled_anchors = 0

    for case in cases:
        experiment.reset()

        # Step 1: Feed dialogue history
        for turn in case.get("dialogue_history", []):
            if turn["speaker"] == "user":
                experiment.run_turn(turn["text"])

        # Step 2: Ask memory questions
        questions = case.get("memory_questions", [])
        recalled_flags: List[bool] = []

        for q_item in questions:
            question = q_item["question"]
            anchor = q_item["anchor"]

            # Ask the experiment to answer the memory question
            response = experiment.run_turn(question)
            recalled = _check_anchor_recalled(response, anchor, question)
            recalled_flags.append(recalled)

        # Step 3: Compute AP for this case
        ap = _average_precision(recalled_flags)
        total_anchors += len(recalled_flags)
        recalled_anchors += sum(recalled_flags)
        all_ap_scores.append(ap)

        per_case_results.append({
            "case_id": case.get("id", f"case_{len(per_case_results)}"),
            "AP": round(ap, 4),
            "n_anchors": len(recalled_flags),
            "recalled": sum(recalled_flags),
            "per_question": [
                {
                    "question": q["question"],
                    "anchor": q["anchor"],
                    "recalled": r,
                }
                for q, r in zip(questions, recalled_flags)
            ],
        })

    MAP = sum(all_ap_scores) / len(all_ap_scores) if all_ap_scores else 0.0

    return {
        "MAP": round(MAP, 4),
        "per_case": per_case_results,
        "total_anchors": total_anchors,
        "recalled_anchors": recalled_anchors,
        "recall_rate": round(recalled_anchors / total_anchors, 4) if total_anchors > 0 else 0.0,
        "n_cases": len(per_case_results),
    }
