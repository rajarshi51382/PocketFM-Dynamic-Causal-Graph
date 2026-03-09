"""
State propagation module.

Responsible for propagating belief updates and event effects
into emotional state, relationships, and intentions.

This module operates on the CharacterState defined in
core.data_structures and modifies internal variables
based on the incoming event frame.

All functions modify CharacterState in place and return the updated
state for convenience.
"""

import math

from core.data_structures import CharacterState, EventFrame, RelationshipState

_EMOTION_TONE_MAP = {
    "joy": ("valence", +0.3),
    "anger": ("valence", -0.3),
    "fear": ("valence", -0.2),
    "sadness": ("valence", -0.25),
    "surprise": ("arousal", +0.3),
    "disgust": ("valence", -0.15),
    "trust": ("valence", +0.1),
    "anticipation": ("arousal", +0.2),
}

_TRUST_TONE_MAP = {
    "joy": +0.05,
    "trust": +0.1,
    "anger": -0.1,
    "disgust": -0.08,
    "fear": -0.05,
    "sadness": -0.03,
}


def update_emotional_state(state: CharacterState, event: EventFrame) -> CharacterState:
    """
    Update the character's emotional state based on the event tone.

    Applies a plasticity-weighted shift to valence and arousal. Discrete
    emotion tag intensities are updated additively and clamped to [0, 1].
    
    Incorporates TraitState:
    - neuroticism: Amplifies all emotional shifts.
    - stoicism: Dampens all emotional shifts.

    Preconditions
    -------------
    state : CharacterState
        Current internal character state
    event : EventFrame
        Structured representation of the incoming dialogue event

    Procedure
    ---------
    1. Read emotional tone (τt) from the event frame
    2. Adjust valence and arousal accordingly
    3. Update emotion tags to reflect detected emotions
    4. Apply emotional plasticity modulated by traits

    Postconditions
    --------------
    state.emotions updated

    Parameters
    ----------
    state : CharacterState
    event : EventFrame

    Returns
    -------
    CharacterState
        The same object, modified in place.
    """
    tone = event.emotional_tone
    if tone is None:
        return state

    tone_lower = tone.strip().lower()
    
    # Calculate effective plasticity based on traits
    base_alpha = state.emotions.plasticity
    neuroticism = state.traits.get("neuroticism", 0.0)
    stoicism = state.traits.get("stoicism", 0.0)
    
    # Neuroticism increases volatility; Stoicism reduces it.
    effective_alpha = _clamp(base_alpha * (1.0 + neuroticism - stoicism), 0.1, 1.0)

    if tone_lower in _EMOTION_TONE_MAP:
        dimension, delta = _EMOTION_TONE_MAP[tone_lower]
        if dimension == "valence":
            state.emotions.valence = _clamp(
                state.emotions.valence + effective_alpha * delta, -1.0, 1.0
            )
        else:
            state.emotions.arousal = _clamp(
                state.emotions.arousal + effective_alpha * delta, 0.0, 1.0
            )

    prev = state.emotions.emotion_tags.get(tone_lower, 0.0)
    state.emotions.emotion_tags[tone_lower] = _clamp(prev + effective_alpha * 0.3, 0.0, 1.0)

    return state


def update_relationship_state(state: CharacterState, event: EventFrame) -> CharacterState:
    """
    Update relationship values between the character and event entities.

    Trust shifts are driven by the event's emotional tone.
    Dynamically creates new relationships for unknown entities.
    
    Incorporates TraitState:
    - trusting: Increases initial trust and positive updates.
    - suspicious: Decreases initial trust and amplifies negative updates.
    - agreeableness: Amplifies positive affection shifts.

    Preconditions
    -------------
    state : CharacterState
    event : EventFrame

    Procedure
    ---------
    1. Identify interacting entities from event.entities
    2. Locate or create relationship nodes in state.relationships
    3. Adjust trust, affection, or respect based on event tone
    4. Apply relationship decay or reinforcement

    Postconditions
    --------------
    state.relationships updated

    Parameters
    ----------
    state : CharacterState
    event : EventFrame

    Returns
    -------
    CharacterState
        The same object, modified in place.
    """
    tone = (event.emotional_tone or "").strip().lower()
    delta_trust = _TRUST_TONE_MAP.get(tone, 0.0)
    
    # Trait modifiers
    trusting = state.traits.get("trusting", 0.0)
    suspicious = state.traits.get("suspicious", 0.0)
    agreeableness = state.traits.get("agreeableness", 0.0)

    for entity in event.entities:
        # Skip self-reference
        if entity == state.character_id:
            continue
            
        rel = state.relationships.get(entity)
        
        # Dynamic discovery: Create new relationship if it doesn't exist
        if rel is None:
            initial_trust = 0.5 + (0.2 * trusting) - (0.2 * suspicious)
            rel = RelationshipState(
                entity_id=entity,
                trust=_clamp(initial_trust, 0.0, 1.0),
                affection=0.5 + (0.1 * agreeableness),
                respect=0.5
            )
            state.add_relationship(rel)

        # Apply update
        # If delta is positive, 'trusting' boosts it. If negative, 'suspicious' boosts it (makes it worse).
        if delta_trust > 0:
            modified_delta = delta_trust * (1.0 + trusting)
        else:
            modified_delta = delta_trust * (1.0 + suspicious)
            
        rel.trust = _clamp(rel.trust + modified_delta * event.confidence, 0.0, 1.0)
        
        # Simple affection update based on tone (simplified)
        if tone in ["joy", "trust", "love"]:
            rel.affection = _clamp(rel.affection + 0.05 * (1.0 + agreeableness), 0.0, 1.0)
        elif tone in ["anger", "disgust", "hate"]:
            rel.affection = _clamp(rel.affection - 0.05, 0.0, 1.0)

    return state


def update_intentions(state: CharacterState) -> CharacterState:
    """
    Derive character intentions from current beliefs and emotional state.
    """
    intentions = []

    # 1. Belief-driven intentions
    for key, belief in state.beliefs.items():
        prob = belief.probability
        if prob > 0.85:
            # If we believe something positive, we want to maintain it
            if "danger" not in key and "locked" not in key:
                intentions.append(f"protect_{key}")
        elif prob < 0.15:
            # If we strongly disbelieve something, we might want to investigate
            intentions.append(f"investigate_{key}")

    # 2. Emotion-driven intentions
    if state.emotions.arousal > 0.7:
        if state.emotions.valence < -0.3:
            intentions.append("confront_threat")
        elif state.emotions.valence > 0.3:
            intentions.append("celebrate_success")
            
    # 3. Trait-driven intentions
    if state.traits.get("curiosity", 0.0) > 0.5:
        intentions.append("explore_surroundings")
    if state.traits.get("bravery", 0.0) < 0.2 and state.emotions.valence < 0:
        intentions.append("seek_reassurance")

    # Limit to top 3 intentions to avoid clutter
    state.intentions = list(dict.fromkeys(intentions))[:3]
    return state


def propagate_state_updates(state: CharacterState, event: EventFrame) -> CharacterState:
    """
    Execute the full internal state update pipeline for one turn.

    Steps:
    1. Update emotional state from event tone.
    2. Update relationship state for referenced entities.
    3. Derive updated intentions from beliefs and emotions.
    4. Increment the timeline index.

    Preconditions
    -------------
    state : CharacterState
    event : EventFrame

    Procedure
    ---------
    1. Update emotional state
    2. Update relationship state
    3. Update intentions
    4. Increment timeline index

    Postconditions
    --------------
    Updated CharacterState returned

    Parameters
    ----------
    state : CharacterState
    event : EventFrame

    Returns
    -------
    CharacterState
        The same object, modified in place.
    """
    update_emotional_state(state, event)
    update_relationship_state(state, event)
    update_intentions(state)
    state.timeline_index += 1
    return state


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))