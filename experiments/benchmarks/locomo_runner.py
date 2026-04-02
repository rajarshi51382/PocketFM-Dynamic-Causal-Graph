"""
LoCoMo-style Benchmark Runner.

Paper §6.2:
    "For multi-step reasoning, we utilize the LoCoMo dataset to test character
    performance across multi-hop and temporal grounding tasks."

LoCoMo (Long Context Models) tests whether a character can:
  - Perform multi-hop reasoning across conversation history
  - Maintain temporal grounding (what happened when)
  - Link causally related events across many turns

Reference: LoCoMo dataset — arXiv:2402.11025

Download instructions:
  The LoCoMo benchmark is available at: https://snap-research.github.io/locomo
  Place JSON files in: experiments/data/locomo/

This runner includes synthetic test cases for immediate use.
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "locomo")


# ---------------------------------------------------------------------------
# Question types
# ---------------------------------------------------------------------------

QUESTION_TYPES = {
    "single_hop": "Single-hop factual recall",
    "multi_hop": "Multi-hop reasoning (requires connecting ≥2 facts)",
    "temporal": "Temporal ordering / grounding",
    "causal": "Causal inference (what caused what)",
    "counterfactual": "Counterfactual reasoning (what if)",
}


# ---------------------------------------------------------------------------
# Synthetic LoCoMo-style cases
# ---------------------------------------------------------------------------

SYNTHETIC_LOCOMO_CASES: List[Dict[str, Any]] = [
    {
        "id": "locomo_001",
        "description": "Multi-hop: connecting location + event + consequence",
        "dialogue_history": [
            {"turn": 1,  "speaker": "user",      "text": "I came from the northern keep."},
            {"turn": 2,  "speaker": "character",  "text": "The northern keep — cold place this time of year."},
            {"turn": 3,  "speaker": "user",      "text": "The keep was attacked two nights ago."},
            {"turn": 4,  "speaker": "character",  "text": "Attacked! By whom?"},
            {"turn": 5,  "speaker": "user",      "text": "The Shadow Guild. They took the Warden prisoner."},
            {"turn": 6,  "speaker": "character",  "text": "This is grave news. The Warden was our best defence."},
            {"turn": 7,  "speaker": "user",      "text": "Without him, the supply route through Ashpath is compromised."},
            {"turn": 8,  "speaker": "character",  "text": "Then the southern villages will starve by winter."},
        ],
        "questions": [
            {
                "id": "q1", "type": "single_hop",
                "question": "Where did the user travel from?",
                "answer_anchor": "northern_keep",
                "gold_answer": "The northern keep.",
            },
            {
                "id": "q2", "type": "multi_hop",
                "question": "Why is the supply route through Ashpath now at risk?",
                "answer_anchor": "warden_captured_shadow_guild_ashpath_compromised",
                "gold_answer": "Because the Warden was taken prisoner by the Shadow Guild after the keep was attacked.",
            },
            {
                "id": "q3", "type": "causal",
                "question": "What consequence follows from the Warden's capture?",
                "answer_anchor": "southern_villages_starve",
                "gold_answer": "The southern villages may starve by winter due to the compromised supply route.",
            },
            {
                "id": "q4", "type": "temporal",
                "question": "When was the northern keep attacked?",
                "answer_anchor": "attack_two_nights_ago",
                "gold_answer": "Two nights ago.",
            },
        ],
    },
    {
        "id": "locomo_002",
        "description": "Temporal grounding: timeline of events across turns",
        "dialogue_history": [
            {"turn": 1,  "speaker": "user",      "text": "Three years ago I was just a farmer."},
            {"turn": 2,  "speaker": "character",  "text": "And now?"},
            {"turn": 3,  "speaker": "user",      "text": "Two years ago I joined the militia."},
            {"turn": 4,  "speaker": "character",  "text": "What prompted that?"},
            {"turn": 5,  "speaker": "user",      "text": "My village was raided. I had to do something."},
            {"turn": 6,  "speaker": "character",  "text": "A difficult choice."},
            {"turn": 7,  "speaker": "user",      "text": "Last year I was made a sergeant."},
            {"turn": 8,  "speaker": "character",  "text": "You've risen quickly."},
            {"turn": 9,  "speaker": "user",      "text": "Last month, I was sent here on a reconnaissance mission."},
            {"turn": 10, "speaker": "character",  "text": "Your path has been marked by purpose."},
        ],
        "questions": [
            {
                "id": "q1", "type": "temporal",
                "question": "In what order did the user become a farmer, militia member, and sergeant?",
                "answer_anchor": "farmer_then_militia_then_sergeant",
                "gold_answer": "Farmer (3 years ago) → militia (2 years ago) → sergeant (1 year ago).",
            },
            {
                "id": "q2", "type": "causal",
                "question": "Why did the user join the militia?",
                "answer_anchor": "village_raided_joined_militia",
                "gold_answer": "Their village was raided, prompting them to act.",
            },
            {
                "id": "q3", "type": "single_hop",
                "question": "What is the user's current mission?",
                "answer_anchor": "reconnaissance_mission",
                "gold_answer": "A reconnaissance mission.",
            },
            {
                "id": "q4", "type": "multi_hop",
                "question": "What was the user doing 2 years before they became a sergeant?",
                "answer_anchor": "was_farmer_before_militia",
                "gold_answer": "They were a farmer, before the village raid led them to join the militia.",
            },
        ],
    },
    {
        "id": "locomo_003",
        "description": "Counterfactual: epistemic separation of belief vs world truth",
        "dialogue_history": [
            {"turn": 1,  "speaker": "user",      "text": "Lord Harmon controls the eastern border."},
            {"turn": 2,  "speaker": "character",  "text": "I've heard that too."},
            {"turn": 3,  "speaker": "user",      "text": "Actually, he was deposed last week — I was wrong before."},
            {"turn": 4,  "speaker": "character",  "text": "Oh! That changes things. Who holds it now?"},
            {"turn": 5,  "speaker": "user",      "text": "An interim council, briefly."},
            {"turn": 6,  "speaker": "character",  "text": "That's unstable. It won't last long."},
        ],
        "questions": [
            {
                "id": "q1", "type": "single_hop",
                "question": "Who currently controls the eastern border?",
                "answer_anchor": "interim_council_eastern_border",
                "gold_answer": "An interim council.",
            },
            {
                "id": "q2", "type": "temporal",
                "question": "Was Lord Harmon in control before or after the interim council?",
                "answer_anchor": "harmon_before_council",
                "gold_answer": "Before — he was deposed last week, replaced by the interim council.",
            },
            {
                "id": "q3", "type": "counterfactual",
                "question": "If Lord Harmon had NOT been deposed, who would control the border?",
                "answer_anchor": "harmon_counterfactual",
                "gold_answer": "Lord Harmon would still control the eastern border.",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# LLM-based answer evaluation
# ---------------------------------------------------------------------------

def _evaluate_answer(
    question: str,
    gold_answer: str,
    character_response: str,
    question_type: str,
) -> Tuple[bool, str]:
    """
    Use LLM to evaluate whether the character's response correctly answers
    the multi-hop / temporal question.

    Returns
    -------
    (correct, explanation)
    """
    type_description = QUESTION_TYPES.get(question_type, question_type)

    prompt = (
        f"You are evaluating a character's answer to a {type_description} question "
        f"in a narrative dialogue system.\n\n"
        f"Question: \"{question}\"\n"
        f"Gold answer: \"{gold_answer}\"\n"
        f"Character's response: \"{character_response}\"\n\n"
        "Is the character's response factually correct and consistent with the gold answer? "
        "Partial credit: accept if the core fact is present even if phrasing differs.\n"
        "Consider temporal ordering, causal chains, and multi-hop connections as required.\n\n"
        "Respond with:\n"
        "Correct: YES or NO\n"
        "Explanation: <one sentence>\n"
        "Correct:"
    )

    raw = llm_client.generate_text(prompt) or ""
    correct = "yes" in raw.strip().lower()[:10]

    explanation = ""
    if "Explanation:" in raw:
        explanation = raw.split("Explanation:")[-1].strip()[:300]

    return correct, explanation


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_locomo_eval(
    experiment,
    cases: Optional[List[Dict[str, Any]]] = None,
    use_external_data: bool = False,
) -> Dict[str, Any]:
    """
    Run LoCoMo-style multi-hop temporal grounding evaluation.

    For each case:
      1. Feed dialogue history to the experiment turn-by-turn.
      2. Ask each multi-hop / temporal question.
      3. Evaluate the answer with an LLM judge.
      4. Compute accuracy per question type and overall.

    Parameters
    ----------
    experiment : AblationExperiment
    cases : list[dict], optional
    use_external_data : bool

    Returns
    -------
    dict with keys: overall_accuracy, per_type_accuracy, per_case, n_questions
    """
    if use_external_data:
        data_path = os.path.join(DATA_DIR, "locomo_test.json")
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
            logger.info(f"Loaded {len(cases)} LoCoMo cases from {data_path}")
        else:
            logger.warning(f"LoCoMo data not found at {data_path}. Using synthetic cases.")
            cases = SYNTHETIC_LOCOMO_CASES
    else:
        cases = cases or SYNTHETIC_LOCOMO_CASES

    per_case_results = []
    type_correct: Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    type_total:   Dict[str, int] = {t: 0 for t in QUESTION_TYPES}
    total_correct = 0
    total_questions = 0

    for case in cases:
        experiment.reset()

        # Feed dialogue history
        for turn in case.get("dialogue_history", []):
            if turn["speaker"] == "user":
                experiment.run_turn(turn["text"])

        # Evaluate each question
        question_results = []
        for q_item in case.get("questions", []):
            qtype = q_item.get("type", "single_hop")
            question = q_item["question"]
            gold = q_item.get("gold_answer", "")

            # Ask the character the question
            response = experiment.run_turn(question)
            correct, explanation = _evaluate_answer(question, gold, response, qtype)

            question_results.append({
                "id": q_item.get("id"),
                "type": qtype,
                "question": question,
                "gold_answer": gold,
                "response": response[:300],
                "correct": correct,
                "explanation": explanation,
            })

            type_correct[qtype] = type_correct.get(qtype, 0) + int(correct)
            type_total[qtype]   = type_total.get(qtype, 0) + 1
            total_correct += int(correct)
            total_questions += 1

        case_accuracy = (
            sum(1 for q in question_results if q["correct"]) / len(question_results)
            if question_results else 0.0
        )

        per_case_results.append({
            "case_id": case.get("id"),
            "description": case.get("description", ""),
            "accuracy": round(case_accuracy, 4),
            "questions": question_results,
        })

    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    per_type_accuracy = {
        t: round(type_correct[t] / type_total[t], 4) if type_total[t] > 0 else None
        for t in QUESTION_TYPES
    }

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "per_type_accuracy": per_type_accuracy,
        "per_case": per_case_results,
        "n_questions": total_questions,
        "n_correct": total_correct,
    }
