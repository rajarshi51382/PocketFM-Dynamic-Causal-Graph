"""
Event extraction pipeline.

Responsible for converting raw user dialogue into structured event frames
used by the update phase.

Converts raw user dialogue into structured EventFrame objects.
In production this module calls an LLM extraction pass followed by a
validation pass. The stubs below provide a rule-based fallback used
in tests and offline demos.
"""

import re
import json
import logging
import math
import os
from functools import lru_cache
from typing import Optional, List, Dict, Any

from core.data_structures import EventFrame
from core import llm_client

logger = logging.getLogger(__name__)

# Fallback tone map used if LLM fails
_TONE_MAP = {
    "joy": ["happy", "good", "great", "joy", "wonderful", "excellent", "thanks", "thank"],
    "anger": ["angry", "mad", "hate", "furious", "annoyed", "stop"],
    "sadness": ["sad", "bad", "unfortunate", "sorry", "regret", "pity"],
    "fear": ["scared", "fear", "afraid", "terrified", "worry", "worried"],
    "trust": ["trust", "believe", "agree", "sure", "certain"],
    "disgust": ["gross", "disgust", "hate", "eww"],
    "surprise": ["wow", "really", "unexpected", "surprise"],
    "anticipation": ["hope", "expect", "wait", "looking forward"]
}

# Embedding labels for OOD matching fallback
_EMBEDDING_LABELS: dict[str, list[str]] = {
    "castle_is_safe": [
        "the castle is safe",
        "the fortress is secure",
        "the stronghold feels protected",
        "the castle is not unsafe",
        "the fortress is not dangerous",
    ],
    "not_castle_is_safe": [
        "the castle is unsafe",
        "the fortress is dangerous",
        "the stronghold is crumbling",
    ],
    "king_is_wise": [
        "the king is wise",
        "the ruler is intelligent",
        "the monarch is trustworthy",
    ],
    "not_king_is_wise": [
        "the king is foolish",
        "the ruler is corrupt",
        "the monarch is unwise",
    ],
    "forest_is_dangerous": [
        "the forest is dangerous",
        "the woods are unsafe",
        "the woodland is threatening",
    ],
    "not_forest_is_dangerous": [
        "the forest is safe",
        "the woods feel calm",
        "the woodland is peaceful",
    ],
    "ally_is_trustworthy": [
        "my ally is completely trustworthy",
        "my friend is reliable",
        "my companion is faithful",
    ],
    "not_ally_is_trustworthy": [
        "my ally is untrustworthy",
        "they betrayed me",
        "my companion is a traitor",
    ],
    "enemy_is_approaching": [
        "the enemy is approaching",
        "an army is coming",
        "there is a threat nearby",
    ],
    "not_enemy_is_approaching": [
        "the enemy is retreating",
        "the threat is gone",
    ],
    "peace_declared": [
        "peace has been declared",
        "the war is over",
    ],
    "not_peace_declared": [
        "the war has begun",
        "peace is broken",
    ]
}

_EMBEDDING_THRESHOLD = float(os.getenv("EMBEDDING_MATCH_THRESHOLD", "0.35"))

def _normalize_base_prop(prop: str) -> str:
    p = prop.strip().lower()
    if p.startswith("not_"):
        return p[4:]
    if p.startswith("~"):
        return p[1:]
    return p

def _canonicalize_proposition(prop: str, allowed_predicates: set[str]) -> str | None:
    """
    Map extracted proposition into the existing belief schema.
    Returns canonical proposition string, usually:
      - king_is_wise
      - not_king_is_wise
    Returns None if no safe mapping is found.
    """
    p = prop.strip().lower()

    if not p:
        return None

    # already in canonical positive form
    base = _normalize_base_prop(p)
    if base in allowed_predicates:
        if p.startswith("not_") or p.startswith("~"):
            return f"not_{base}"
        return base

    # lightweight antonym / alias mapping for current MVP
    alias_to_canonical = {
        "king_is_evil": "not_king_is_wise",
        "king_is_bad": "not_king_is_wise",
        "king_is_foolish": "not_king_is_wise",
        "king_is_liar": "not_king_is_wise",
        
        "king_evil": "not_king_is_wise",
        "king_bad": "not_king_is_wise",
        "king_foolish": "not_king_is_wise",
        "king_liar": "not_king_is_wise",
        
        "castle_is_dangerous": "not_castle_is_safe",
        "castle_is_unsafe": "not_castle_is_safe",
        "fortress_is_crumbling": "not_castle_is_safe",
        
        "castle_dangerous": "not_castle_is_safe",
        "castle_unsafe": "not_castle_is_safe",
        "fortress_crumbling": "not_castle_is_safe",
    }

    if p in alias_to_canonical:
        mapped = alias_to_canonical[p]
        mapped_base = _normalize_base_prop(mapped)
        if mapped_base in allowed_predicates:
            return mapped

    return None

def extract_event(user_message: str) -> EventFrame:
    """
    Convert raw dialogue into a structured event frame.

    Attempts to use the configured LLM client. Falls back to a rule-based
    heuristic if the LLM is unavailable or fails.

    Preconditions
    -------------
    user_message : str

    Procedure
    ---------
    1. Check for configured LLM client.
    2. If available, prompt for structured extraction.
    3. If unavailable/fails, use regex/heuristic fallback.

    Postconditions
    --------------
    Returns EventFrame representing the dialogue event

    Parameters
    ----------
    user_message : str

    Returns
    -------
    EventFrame
    """
    if llm_client.configure_client() and llm_client.is_embedding_available():
        try:
            return _extract_event_pure_embeddings(user_message)
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise RuntimeError("Event extraction requires a valid Gemini API key for embeddings.") from e
            
    raise RuntimeError("Event extraction requires a valid Gemini API key for embeddings. Please set GEMINI_API_KEY in the sidebar.")





def _extract_event_pure_embeddings(user_message: str) -> EventFrame:
    """Implement fully embedding-based event extraction (propositions and tone)."""
    message = user_message.strip().lower()
    
    message_embedding = llm_client.get_embedding(message)
    if not message_embedding:
        raise ValueError("Could not retrive embedding for message")

    # 1. Proposition matching via embeddings
    propositions = []
    best_prop = None
    best_score = -1.0
    for prop, labels in _EMBEDDING_LABELS.items():
        for label in labels:
            label_embedding = _get_label_embedding(label)
            if not label_embedding:
                continue
            score = _cosine_similarity(message_embedding, label_embedding)
            if score > best_score:
                best_score = score
                best_prop = prop

    if best_prop is not None and best_score >= _EMBEDDING_THRESHOLD:
        propositions.append(best_prop)

    # 2. Emotional tone matching via embeddings
    best_emo = "neutral"
    best_emo_score = -1.0
    for tone in _TONE_MAP.keys():
        emo_label = f"The emotional tone is {tone}"
        emo_emb = _get_label_embedding(emo_label)
        if emo_emb:
            score = _cosine_similarity(message_embedding, emo_emb)
            if score > best_emo_score:
                best_emo_score = score
                best_emo = tone
                
    # 3. Entity extraction (fallback to regex since embeddings can't extract specific nouns)
    entities = list(set(re.findall(r'\b(?:[A-Z][a-z]+(?:_[A-Z][a-z]+)*|[A-Z]{2,})\b', user_message)))
    
    return EventFrame(
        propositions=propositions,
        entities=entities,
        speaker="user",
        emotional_tone=best_emo,
        confidence=1.0 if propositions else 0.5,
        source_text=user_message
    )


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity for two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return -1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return dot / (norm_a * norm_b)


@lru_cache(maxsize=256)
def _get_label_embedding(label: str) -> list[float] | None:
    """Cache embeddings for label phrases."""
    return llm_client.get_embedding(label)


def validate_event(
    event: EventFrame,
    user_message: str,
    allowed_predicates: set[str] | None = None,
) -> EventFrame:
    """
    Validate extraction output and adjust confidence.

    Checks that the event frame has at least one proposition and
    that the confidence is within bounds. Downstream callers should
    treat a returned confidence of 0.0 as an extraction failure.

    Preconditions
    -------------
    event : EventFrame
    user_message : str

    Procedure
    ---------
    1. Check if propositions were extracted
    2. Adjust confidence based on message length vs extraction
    3. Clamp confidence to [0, 1]

    Postconditions
    --------------
    Returns corrected EventFrame

    Parameters
    ----------
    event : EventFrame
    user_message : str

    Returns
    -------
    EventFrame
        Corrected or unchanged EventFrame.
    """
    # If no propositions and no entities, it's a weak extraction
    if not event.propositions and not event.entities:
        event.confidence *= 0.1
    
    # If the message is very long but only one short prop was extracted, lower confidence
    if len(user_message.split()) > 10 and len(event.propositions) == 1:
        event.confidence *= 0.8

    if allowed_predicates is not None:
        canonical_props = []
        for prop in event.propositions:
            mapped = _canonicalize_proposition(prop, allowed_predicates)
            if mapped is not None:
                canonical_props.append(mapped)
        
        # dedupe while preserving order
        seen = set()
        event.propositions = [
            p for p in canonical_props
            if not (p in seen or seen.add(p))
        ]
    
    # Final clamping
    event.confidence = max(0.0, min(1.0, event.confidence))
    
    # If confidence is extremely low, mark as 0
    if event.confidence < 0.1:
        event.confidence = 0.0
        
    return event