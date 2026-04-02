"""
Ablation variant registry.

Maps paper variant labels (A–F) and human-readable names to their
experiment classes so the CLI runner can instantiate them by name.
"""

from .baseline_lm import BaselineLMExperiment
from .state_only import StateOnlyExperiment
from .state_planning import StatePlanningExperiment
from .state_verification import StateVerificationExperiment
from .full_pipeline import FullPipelineExperiment
from .critic_only import CriticOnlyExperiment

# Paper §6 ablation label → class
VARIANT_REGISTRY = {
    "A": BaselineLMExperiment,
    "B": StateOnlyExperiment,
    "C": StatePlanningExperiment,
    "D": StateVerificationExperiment,
    "E": FullPipelineExperiment,
    "F": CriticOnlyExperiment,
}

# Human-readable aliases
VARIANT_ALIASES = {
    "baseline_lm":        "A",
    "state_only":         "B",
    "state_planning":     "C",
    "state_verification": "D",
    "full_pipeline":      "E",
    "critic_only":        "F",
}

VARIANT_DESCRIPTIONS = {
    "A": "Baseline LM — context-only generation, no structured state.",
    "B": "State-Only (DCCG) — Two-phase Update–Act loop, direct generation.",
    "C": "State + Planning — Structured state with intermediate plan node.",
    "D": "State + Verification — Structured state with post-generation verification.",
    "E": "Full Pipeline — State + Planning + Verification (complete DCCG).",
    "F": "Critic-Only Control — Context-only baseline + post-hoc verification.",
}


def get_variant(label: str) -> type:
    """Return the experiment class for a given variant label or alias."""
    label = label.strip()
    if label in VARIANT_ALIASES:
        label = VARIANT_ALIASES[label]
    if label not in VARIANT_REGISTRY:
        raise KeyError(
            f"Unknown variant '{label}'. "
            f"Valid options: {list(VARIANT_REGISTRY.keys()) + list(VARIANT_ALIASES.keys())}"
        )
    return VARIANT_REGISTRY[label]


__all__ = [
    "VARIANT_REGISTRY",
    "VARIANT_ALIASES",
    "VARIANT_DESCRIPTIONS",
    "get_variant",
    "BaselineLMExperiment",
    "StateOnlyExperiment",
    "StatePlanningExperiment",
    "StateVerificationExperiment",
    "FullPipelineExperiment",
    "CriticOnlyExperiment",
]
