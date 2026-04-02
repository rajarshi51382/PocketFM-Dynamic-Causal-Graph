"""
Metrics package for DCCG ablation evaluation.

Paper §5 defines four evaluation dimensions:
  - Personality Stability    → drift_metrics.py    (UDR, FAR)
  - Behavioral Grounding     → grounding_metrics.py (G score)
  - Closed-loop Hallucination→ hallucination_metrics.py (H rate)
  - Narrative Consistency    → narrative_metrics.py  (KnowledgeLeak, NC score)
"""

from .drift_metrics import compute_udr, compute_far, compute_sigma
from .grounding_metrics import compute_grounding_score
from .hallucination_metrics import compute_hallucination_rate
from .narrative_metrics import compute_knowledge_leak, compute_narrative_consistency

__all__ = [
    "compute_udr",
    "compute_far",
    "compute_sigma",
    "compute_grounding_score",
    "compute_hallucination_rate",
    "compute_knowledge_leak",
    "compute_narrative_consistency",
]
