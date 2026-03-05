"""
Defines the shared schemas used throughout the system.

These structures represent the causal character graph state,
world state, and structured event frames.
"""


class TraitState:
    """
    Represents stable personality traits.

    Attributes
    ----------
    traits : dict
        Mapping of trait name to intensity.

    plasticity : float
        Controls how slowly traits evolve.
    """

    def __init__(self, traits: dict, plasticity: float):
        # INCOMPLETE
        pass


class EmotionState:
    """
    Represents the emotional state of a character.

    Attributes
    ----------
    valence : float
    arousal : float
    emotion_tags : dict
    plasticity : float
    """

    def __init__(self):
        # INCOMPLETE
        pass


class RelationshipState:
    """
    Represents relationship values with another entity.

    Attributes
    ----------
    trust : float
    affection : float
    respect : float
    """

    def __init__(self):
        # INCOMPLETE
        pass


class BeliefNode:
    """
    Represents a belief proposition using log-odds.

    Attributes
    ----------
    proposition : str
    log_odds : float
    evidence_sources : list
    """

    def __init__(self, proposition: str, log_odds: float):
        # INCOMPLETE
        pass


class CharacterState:
    """
    Full internal character state.

    Attributes
    ----------
    traits
    emotions
    beliefs
    relationships
    intentions
    timeline_index
    """

    def __init__(self):
        # INCOMPLETE
        pass


class WorldState:
    """
    Represents objective world state.

    Attributes
    ----------
    entities
    object_states
    constraints
    timeline_index
    """

    def __init__(self):
        # INCOMPLETE
        pass


class EventFrame:
    """
    Structured event representation.

    et = (Pt, Et, at, τt, ct)

    Attributes
    ----------
    propositions (Pt)
    entities (Et)
    speaker (at)
    emotional tone (τt)
    confidence (ct)
    """

    def __init__(self):
        # INCOMPLETE
        pass