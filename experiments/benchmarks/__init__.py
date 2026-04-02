"""
Benchmarks package for DCCG evaluation.

Implements runners for external benchmark protocols referenced in paper §6.2:
  - PerLTQA : MAP-style memory anchor recall evaluation
  - LoCoMo  : Multi-hop temporal grounding evaluation
"""

from .perltqa_runner import run_perltqa_eval
from .locomo_runner import run_locomo_eval

__all__ = ["run_perltqa_eval", "run_locomo_eval"]
