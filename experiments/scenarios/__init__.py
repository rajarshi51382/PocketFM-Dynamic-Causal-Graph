"""
Scenarios package for DCCG ablation experiments.

Three scenario types based on paper §5.1:
  - Stability scenarios  : low-surprisal, no character change required
  - Growth scenarios     : high-impact shock events, adaptation required
  - Hallucination scenarios: forced false-claim injection sequences
"""

from .stability_scenarios import STABILITY_SCENARIOS, ScenarioTurn
from .growth_scenarios import GROWTH_SCENARIOS
from .hallucination_scenarios import HALLUCINATION_SCENARIOS

__all__ = [
    "ScenarioTurn",
    "STABILITY_SCENARIOS",
    "GROWTH_SCENARIOS",
    "HALLUCINATION_SCENARIOS",
]
