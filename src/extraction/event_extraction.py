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
    # Attempt LLM extraction first
    if llm_client.configure_client():
        try:
            return _extract_event_llm(user_message)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}. Falling back to rules.")
    
    # Fallback to rule-based extraction
    return _extract_event_rules(user_message)


def _extract_event_llm(user_message: str) -> EventFrame:
    """Helper for LLM-based extraction."""
    schema = {
        "type": "object",
        "properties": {
            "propositions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of factual assertions in snake_case (e.g. 'door_is_locked', 'not_safe'). Use 'not_' prefix for negation."
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of proper nouns or entities mentioned."
            },
            "emotional_tone": {
                "type": "string",
                "enum": ["joy", "anger", "sadness", "fear", "trust", "disgust", "surprise", "anticipation", "neutral"],
                "description": "Dominant emotional tone of the speaker."
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the extraction (0.0 to 1.0)."
            }
        },
        "required": ["propositions", "entities", "emotional_tone", "confidence"]
    }

    prompt = (
        f"Analyze the following dialogue line from a user in a roleplay scenario.\n"
        f"Extract factual propositions (snake_case), mentioned entities, and emotional tone.\n"
        f"User Message: \"{user_message}\"\n"
    )

    result = llm_client.generate_structured(prompt, schema)
    
    if not result:
        raise ValueError("LLM returned empty result")

    return EventFrame(
        propositions=result.get("propositions", []),
        entities=result.get("entities", []),
        speaker="user",
        emotional_tone=result.get("emotional_tone", "neutral"),
        confidence=float(result.get("confidence", 0.5)),
        source_text=user_message
    )


def _extract_simple_propositions(message: str) -> list[str]:
    """
    Simple fallback extraction when semantic patterns don't match.
    Extracts basic subject_predicate patterns from the message.
    """
    propositions = []
    clauses = re.split(r'[.!?;,]', message)
    
    for clause in clauses:
        clause = clause.strip().lower()
        if not clause:
            continue
            
        words = clause.split()
        is_negated = any(neg in words for neg in ["not", "never", "no", "neither", "nor"])
        
        # Remove common stopwords
        stopwords = {"not", "never", "no", "a", "an", "the", "is", "are", "was", "were", 
                     "has", "have", "had", "be", "been", "being", "i", "you", "he", "she",
                     "it", "we", "they", "that", "this", "and", "or", "but", "if", "then",
                     "so", "very", "just", "really", "actually", "heard", "think", "believe"}
        clean_words = [w for w in words if w not in stopwords and len(w) > 1]
        
        if len(clean_words) >= 2:
            # Try to form a simple proposition from first noun + adjective/verb
            prop = "_".join(clean_words[:3])  # Limit to 3 words max
            if prop:
                if is_negated:
                    propositions.append(f"not_{prop}")
                else:
                    propositions.append(prop)
    
    return propositions


def _extract_event_rules(user_message: str) -> EventFrame:
    """
    Rule-based fallback logic with keyword pattern matching.
    
    Uses semantic keyword patterns to extract meaningful propositions
    that map to the belief schema, rather than simple word concatenation.
    """
    message = user_message.strip().lower()
    
    # Semantic keyword patterns -> proposition mappings
    # Each pattern is (keywords_any, keywords_all, negation_keywords, proposition)
    # keywords_any: if ANY of these appear, consider the pattern
    # keywords_all: ALL of these must appear (can be empty)
    # negation_keywords: if ANY of these appear, negate the proposition
    # proposition: the base proposition to emit
    
    SEMANTIC_PATTERNS = [
        # Castle safety patterns
        (["castle", "fortress", "walls", "stronghold"], [], 
         ["unsafe", "dangerous", "crumbling", "falling", "broken", "weak", "not safe", "no longer safe", "isn't safe", "not secure"],
         "castle_is_safe", True),  # True = negation_keywords indicate NEGATIVE proposition
        (["castle", "fortress", "walls", "stronghold"], [],
         ["safe", "secure", "strong", "protected", "solid", "sturdy"],
         "castle_is_safe", False),  # False = these keywords indicate POSITIVE proposition
        
        # King wisdom patterns
        (["king", "ruler", "monarch", "majesty"], [],
         ["wise", "smart", "intelligent", "good", "just", "fair", "trustworthy", "honest"],
         "king_is_wise", False),
        (["king", "ruler", "monarch", "majesty"], [],
         ["foolish", "stupid", "evil", "bad", "liar", "betrayed", "betrayal", "corrupt", "dishonest", "unwise"],
         "king_is_wise", True),
        
        # Forest danger patterns
        (["forest", "woods", "woodland"], [],
         ["dangerous", "unsafe", "scary", "dark", "haunted", "monsters", "beasts", "threat"],
         "forest_is_dangerous", False),
        (["forest", "woods", "woodland"], [],
         ["safe", "clear", "cleared", "peaceful", "calm", "secure"],
         "forest_is_dangerous", True),
        
        # Trust patterns
        (["ally", "friend", "companion"], [],
         ["trustworthy", "trust", "reliable", "loyal", "honest", "faithful"],
         "ally_is_trustworthy", False),
        (["ally", "friend", "companion"], [],
         ["untrustworthy", "betray", "betrayed", "traitor", "dishonest", "liar"],
         "ally_is_trustworthy", True),
        
        # Enemy patterns  
        (["enemy", "enemies", "foe", "army", "invader"], [],
         ["approaching", "coming", "attack", "threat", "danger", "near"],
         "enemy_is_approaching", False),
        
        # Peace/war patterns
        (["peace", "war"], [],
         ["declared", "over", "ended", "armistice", "treaty"],
         "peace_declared", False),
    ]
    
    propositions = []
    
    # Check for explicit negation in the message
    has_negation = any(neg in message for neg in ["not ", "never ", "no ", "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "n't"])
    
    for keywords_any, keywords_all, indicator_keywords, base_prop, indicators_mean_negative in SEMANTIC_PATTERNS:
        # Check if any primary keyword is present
        if not any(kw in message for kw in keywords_any):
            continue
            
        # Check if all required keywords are present
        if keywords_all and not all(kw in message for kw in keywords_all):
            continue
            
        # Check if indicator keywords are present
        indicator_found = any(kw in message for kw in indicator_keywords)
        
        if indicator_found:
            # Determine if this should be positive or negative
            if indicators_mean_negative:
                # The indicator keywords suggest the negative form
                # But check for double negation (e.g., "not unsafe" = safe)
                if has_negation and any(neg_kw in message for neg_kw in ["not unsafe", "not dangerous", "isn't dangerous", "not foolish", "not evil"]):
                    propositions.append(base_prop)
                else:
                    propositions.append(f"not_{base_prop}")
            else:
                # The indicator keywords suggest the positive form
                if has_negation:
                    propositions.append(f"not_{base_prop}")
                else:
                    propositions.append(base_prop)
    
    # Deduplicate while preserving order
    seen = set()
    propositions = [p for p in propositions if not (p in seen or seen.add(p))]
    
    # If no semantic patterns matched, fall back to simple extraction
    if not propositions:
        propositions = _extract_simple_propositions(message)

    # 3. Entity extraction - use original message for proper case detection
    entities = list(set(re.findall(r'\b(?:[A-Z][a-z]+(?:_[A-Z][a-z]+)*|[A-Z]{2,})\b', user_message)))
    
    # 4. Tone detection (message is already lowercase)
    detected_tone = "neutral"
    for tone, keywords in _TONE_MAP.items():
        if any(word in message for word in keywords):
            detected_tone = tone
            break

    return EventFrame(
        propositions=propositions,
        entities=entities,
        speaker="user",
        emotional_tone=detected_tone,
        confidence=1.0 if propositions else 0.5,
        source_text=user_message
    )


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