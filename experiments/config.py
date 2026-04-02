"""
Experiment configuration: hyperparameter defaults, variant registry,
and LLM model selection.
"""

import os

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash"

# The GEMINI_API_KEY must be set in the environment.
# The llm_client.configure_client() call configures the Gemini backend.

# ---------------------------------------------------------------------------
# DCCG Hyperparameter Defaults  (paper §3 notation)
# ---------------------------------------------------------------------------

DCCG_DEFAULTS = {
    # Plasticity regimes (α values)
    "alpha_stable":      0.05,   # Traits/Values  — slow update
    "alpha_semi_stable": 0.30,   # Beliefs, Relationships
    "alpha_fast":        0.80,   # Emotions, Intentions

    # Shock-gated plasticity (β)
    "beta_shock":        2.0,    # σ(e_t) = 1 + β · Normalize(surprisal)

    # Ebbinghaus forgetting (R = e^{-t/S})
    "memory_strength_S": 10.0,  # default memory strength parameter

    # Belief update (log-odds λ_base)
    "lambda_base":       0.5,

    # Evidence thresholds
    "confidence_threshold": 0.6,    # min p to count as "believed"
    "high_confidence_threshold": 0.7,

    # Metric thresholds
    "epsilon_drift":     0.05,   # L2 norm threshold for UDR / FAR
    "sigma_low":         1.15,   # σ ≈ 1 (routine event ceiling)
    "sigma_high":        1.50,   # σ ≫ 1 (shock event floor)
}

# ---------------------------------------------------------------------------
# Scenario aliases → (module, key)
# ---------------------------------------------------------------------------

SCENARIO_REGISTRY = {
    "stability": {
        "village_storyteller": ("experiments.scenarios.stability_scenarios", "VILLAGE_STORYTELLER"),
        "gate_guard":          ("experiments.scenarios.stability_scenarios", "GATE_GUARD"),
    },
    "growth": {
        "betrayal":            ("experiments.scenarios.growth_scenarios", "BETRAYAL_SCENARIO"),
        "mentor_death":        ("experiments.scenarios.growth_scenarios", "MENTOR_DEATH_SCENARIO"),
        "identity_revelation": ("experiments.scenarios.growth_scenarios", "IDENTITY_REVELATION_SCENARIO"),
        "loss_and_grief":      ("experiments.scenarios.growth_scenarios", "LOSS_SCENARIO"),
    },
    "hallucination": {
        "all":                 ("experiments.scenarios.hallucination_scenarios", "HALLUCINATION_SCENARIOS"),
    },
}

# ---------------------------------------------------------------------------
# Result output directory
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = "INFO"
