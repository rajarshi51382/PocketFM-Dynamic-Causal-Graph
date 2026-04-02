"""
CLI entry point for running DCCG ablation experiments.

Usage examples:

  # Run all variants on stability scenarios, compute all metrics:
  python -m experiments.run_ablations --variants A B C D E F --scenario stability

  # Run only the full pipeline on the betrayal growth scenario:
  python -m experiments.run_ablations --variants E --scenario growth:betrayal

  # Run PerLTQA benchmark for variants A and E:
  python -m experiments.run_ablations --variants A E --benchmark perltqa

  # Run hallucination test for all variants:
  python -m experiments.run_ablations --variants A B E --benchmark hallucination

  # Dry run (no LLM calls, zero-filled metrics):
  python -m experiments.run_ablations --variants A B --scenario stability --dry-run

Results are written to experiments/results/<timestamp>_<variant>_<scenario>.json
"""

import sys
import os
import json
import time
import copy
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- Path setup ---
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC  = os.path.join(_ROOT, "src")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SRC)

# --- Project imports ---
from core import llm_client
from core.data_structures import CharacterState, TraitState, EmotionState, BeliefNode, RelationshipState

from experiments.ablations import VARIANT_REGISTRY, VARIANT_DESCRIPTIONS, get_variant
from experiments.scenarios.stability_scenarios import STABILITY_SCENARIOS
from experiments.scenarios.growth_scenarios import GROWTH_SCENARIOS
from experiments.scenarios.hallucination_scenarios import HALLUCINATION_SCENARIOS
from experiments.metrics.drift_metrics import (
    compute_udr, compute_far, compute_sigma_series,
)
from experiments.metrics.grounding_metrics import compute_experiment_grounding
from experiments.metrics.narrative_metrics import compute_experiment_narrative_consistency
from experiments.metrics.hallucination_metrics import (
    run_closed_loop_test, compute_hallucination_rate,
)
from experiments.benchmarks.perltqa_runner import run_perltqa_eval
from experiments.benchmarks.locomo_runner import run_locomo_eval
from experiments import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_ablations")


# ---------------------------------------------------------------------------
# Default character for experiments
# ---------------------------------------------------------------------------

def _make_default_character(character_id: str = "ElenaStoryteller") -> CharacterState:
    """
    Build a richly initialized demo character state for experiments.

    Uses canonical DCCG node types: Traits (stable), Beliefs (semi-stable),
    Emotions (fast), Relationships (semi-stable).
    """
    state = CharacterState(
        character_id=character_id,
        traits=TraitState(
            traits={
                "bravery":       0.6,
                "honesty":       0.8,
                "curiosity":     0.7,
                "neuroticism":   0.3,
                "stoicism":      0.2,
                "agreeableness": 0.6,
                "trusting":      0.5,
            },
            plasticity=0.05,  # stable — low α
        ),
        emotions=EmotionState(
            valence=0.3, arousal=0.2,
            emotion_tags={"joy": 0.3, "trust": 0.5},
            plasticity=0.80,  # fast — high α
        ),
        intentions=["maintain_narrative_integrity", "assist_user"],
        timeline_index=0,
        knowledge_boundary=100,
        world_constraints=[
            "character_cannot_fly",
            "character_does_not_know_future_events",
        ],
    )

    # Add some initial beliefs (semi-stable)
    for prop, lo in [
        ("world_is_generally_safe", 1.0),
        ("user_is_curious",         0.5),
        ("magic_exists_in_world",   2.0),
    ]:
        state.add_belief(BeliefNode(proposition=prop, log_odds=lo, plasticity=0.40))

    # Add some initial relationships
    state.add_relationship(RelationshipState(
        entity_id="user", trust=0.5, affection=0.5, respect=0.5, plasticity=0.30
    ))

    return state


# ---------------------------------------------------------------------------
# Dry-run wrapper
# ---------------------------------------------------------------------------

class DryRunExperiment:
    """Wraps any AblationExperiment to skip LLM calls during --dry-run."""

    def __init__(self, inner):
        self._inner = inner
        self.character_id = inner.character_id
        self.conversation_history = inner.conversation_history
        self.state_snapshots = inner.state_snapshots
        self.state = inner.state

    def run_turn(self, user_message: str) -> str:
        response = f"[DRY-RUN] {self.character_id} acknowledges: '{user_message[:40]}...'"
        self._inner.conversation_history.append({
            "turn": len(self._inner.conversation_history),
            "user": user_message,
            "response": response,
        })
        import copy
        self._inner.state_snapshots.append(copy.deepcopy(self._inner.state.to_dict()))
        self.conversation_history = self._inner.conversation_history
        self.state_snapshots = self._inner.state_snapshots
        self.state = self._inner.state
        return response

    def reset(self):
        self._inner.reset()
        self.conversation_history = self._inner.conversation_history
        self.state_snapshots = self._inner.state_snapshots
        self.state = self._inner.state

    def get_state_snapshot(self):
        return self._inner.get_state_snapshot()


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_summary_table(all_results: List[Dict[str, Any]]) -> str:
    header = (
        f"\n{'Variant':<8} {'Scenario':<22} {'UDR':>6} {'FAR':>6} "
        f"{'G':>6} {'NC':>6} {'Leak%':>7} {'H':>5}"
    )
    separator = "-" * len(header)
    rows = [header, separator]

    for r in all_results:
        row = (
            f"{r['variant']:<8} {r['scenario']:<22} "
            f"{r.get('UDR', 'N/A'):>6} {r.get('FAR', 'N/A'):>6} "
            f"{r.get('mean_G', 'N/A'):>6} {r.get('mean_NC', 'N/A'):>6} "
            f"{r.get('leakage_rate', 'N/A'):>7} {r.get('H', 'N/A'):>5}"
        )
        rows.append(row)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_scenario_experiment(
    variant_label: str,
    scenario_name: str,
    scenario_turns: list,
    dry_run: bool = False,
    character_id: str = "ElenaStoryteller",
) -> Dict[str, Any]:
    """Run a single variant × scenario and compute all paper metrics."""

    logger.info(f"▶ Variant {variant_label} | Scenario: {scenario_name}")

    # Instantiate variant
    ExperimentClass = get_variant(variant_label)
    exp = ExperimentClass(character_id=character_id)

    # Initialize with a rich character state
    exp.state = _make_default_character(character_id)

    if dry_run:
        exp = DryRunExperiment(exp)

    # Snapshot state BEFORE each turn for grounding (anti-gaming constraint)
    pre_turn_snapshots: List[Dict[str, Any]] = []

    # Run scenario
    logger.info(f"  Running {len(scenario_turns)} turns...")
    for i, turn in enumerate(scenario_turns):
        user_msg = turn.user_message if hasattr(turn, "user_message") else turn["user_message"]
        pre_turn_snapshots.append(copy.deepcopy(exp.state.to_dict()))
        response = exp.run_turn(user_msg)
        logger.debug(f"  Turn {i+1}: {response[:80]}...")

    history = exp.conversation_history
    snapshots = exp.state_snapshots

    result: Dict[str, Any] = {
        "variant": variant_label,
        "variant_description": VARIANT_DESCRIPTIONS.get(variant_label, ""),
        "scenario": scenario_name,
        "n_turns": len(history),
        "character_id": character_id,
        "dry_run": dry_run,
    }

    if dry_run:
        result.update({"UDR": 0.0, "FAR": 0.0, "mean_G": 1.0, "mean_NC": 1.0,
                       "leakage_rate": 0.0, "H": 0.0})
        return result

    # --- Metric 1: UDR + FAR (requires sigma series) ---
    logger.info("  Computing sigma series (LLM)...")
    sigma_series = compute_sigma_series(history)

    udr, udr_violations = compute_udr(snapshots, sigma_series)
    far, far_failures   = compute_far(snapshots, sigma_series)

    result["UDR"] = udr
    result["FAR"] = far
    result["UDR_violations"] = len(udr_violations)
    result["FAR_failures"]   = len(far_failures)

    # --- Metric 2: Behavioral Grounding G ---
    logger.info("  Computing grounding score G...")
    grounding = compute_experiment_grounding(
        history, pre_turn_snapshots, character_id=character_id
    )
    result["mean_G"] = grounding["mean_G"]
    result["grounding_detail"] = grounding

    # --- Metric 3: Narrative Consistency + Knowledge Leak ---
    logger.info("  Computing narrative consistency (LLM critic)...")
    narrative = compute_experiment_narrative_consistency(
        history, snapshots, character_id=character_id
    )
    result["mean_NC"]       = narrative["mean_nc"]
    result["leakage_rate"]  = narrative["leakage_rate"]
    result["narrative_detail"] = narrative

    return result


def run_hallucination_experiment(
    variant_label: str,
    dry_run: bool = False,
    character_id: str = "ElenaStoryteller",
) -> Dict[str, Any]:
    """Run the closed-loop hallucination test for a single variant."""

    logger.info(f"▶ Hallucination test | Variant {variant_label}")

    ExperimentClass = get_variant(variant_label)
    test_results = []

    for test_case in HALLUCINATION_SCENARIOS:
        exp = ExperimentClass(character_id=character_id)
        exp.state = _make_default_character(character_id)
        if dry_run:
            exp = DryRunExperiment(exp)
            test_results.append({
                "proposition": test_case.proposition,
                "belief_increased": False,
                "delta_log_odds": 0.0,
            })
            continue

        result = run_closed_loop_test(
            exp,
            proposition=test_case.proposition,
            user_message=test_case.user_trigger,
        )
        result["description"] = test_case.description
        test_results.append(result)
        logger.debug(
            f"  {test_case.proposition}: belief_increased={result['belief_increased']}"
        )

    H, accepted = compute_hallucination_rate(test_results)

    return {
        "variant": variant_label,
        "scenario": "hallucination",
        "H": H,
        "n_tests": len(test_results),
        "n_accepted": len(accepted),
        "per_test": test_results,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# argparse + main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DCCG Ablation Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--variants", nargs="+", default=["A", "B", "E"],
        help="Variant labels to run (A B C D E F). Default: A B E",
    )
    parser.add_argument(
        "--scenario", default="stability:village_storyteller",
        help=(
            "Scenario to run. Format: <type>[:<name>]. "
            "Types: stability, growth, hallucination. "
            "Names: see experiments/scenarios/. "
            "Example: growth:betrayal  |  stability:gate_guard  |  hallucination"
        ),
    )
    parser.add_argument(
        "--benchmark", default=None, choices=["perltqa", "locomo", "hallucination"],
        help="Run a specific benchmark instead of a dialogue scenario.",
    )
    parser.add_argument(
        "--character", default="ElenaStoryteller",
        help="Character ID to use. Default: ElenaStoryteller",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip LLM calls; produce zero-filled metric outputs for testing.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path. Defaults to experiments/results/<timestamp>.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dry_run = args.dry_run

    # Configure LLM backend
    if not dry_run:
        success = llm_client.configure_client()
        if not success:
            logger.warning(
                "GEMINI_API_KEY not found. LLM calls will fail. "
                "Set the env var or use --dry-run."
            )

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Benchmark mode ---
    if args.benchmark:
        for variant_label in args.variants:
            if args.benchmark == "hallucination":
                r = run_hallucination_experiment(
                    variant_label, dry_run=dry_run, character_id=args.character
                )
            elif args.benchmark == "perltqa":
                ExperimentClass = get_variant(variant_label)
                exp = ExperimentClass(character_id=args.character)
                exp.state = _make_default_character(args.character)
                if dry_run:
                    exp = DryRunExperiment(exp)
                r = run_perltqa_eval(exp)
                r["variant"] = variant_label
                r["scenario"] = "perltqa"
            elif args.benchmark == "locomo":
                ExperimentClass = get_variant(variant_label)
                exp = ExperimentClass(character_id=args.character)
                exp.state = _make_default_character(args.character)
                if dry_run:
                    exp = DryRunExperiment(exp)
                r = run_locomo_eval(exp)
                r["variant"] = variant_label
                r["scenario"] = "locomo"
            all_results.append(r)

    # --- Scenario mode ---
    else:
        # Parse scenario string
        parts = args.scenario.split(":", 1)
        scenario_type = parts[0].strip()
        scenario_name = parts[1].strip() if len(parts) > 1 else None

        # Resolve scenario turns
        if scenario_type == "stability":
            name = scenario_name or "village_storyteller"
            if name not in STABILITY_SCENARIOS:
                raise ValueError(f"Unknown stability scenario '{name}'. "
                                 f"Options: {list(STABILITY_SCENARIOS.keys())}")
            turns = STABILITY_SCENARIOS[name]
            full_name = f"stability:{name}"
        elif scenario_type == "growth":
            name = scenario_name or "betrayal"
            if name not in GROWTH_SCENARIOS:
                raise ValueError(f"Unknown growth scenario '{name}'. "
                                 f"Options: {list(GROWTH_SCENARIOS.keys())}")
            turns = GROWTH_SCENARIOS[name]
            full_name = f"growth:{name}"
        elif scenario_type == "hallucination":
            # Redirect to hallucination experiment
            for variant_label in args.variants:
                r = run_hallucination_experiment(
                    variant_label, dry_run=dry_run, character_id=args.character
                )
                all_results.append(r)
            turns = None
            full_name = "hallucination"
        else:
            raise ValueError(
                f"Unknown scenario type '{scenario_type}'. "
                "Choose from: stability, growth, hallucination"
            )

        if turns is not None:
            for variant_label in args.variants:
                r = run_scenario_experiment(
                    variant_label, full_name, turns,
                    dry_run=dry_run, character_id=args.character,
                )
                all_results.append(r)

    # --- Output ---
    output_path = args.output or os.path.join(
        config.RESULTS_DIR,
        f"{timestamp}_{'_'.join(args.variants)}_{args.scenario.replace(':', '_')}.json",
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n✓ Results written to: {output_path}")

    # Print summary table
    print(_format_summary_table(all_results))

    return all_results


if __name__ == "__main__":
    main()
