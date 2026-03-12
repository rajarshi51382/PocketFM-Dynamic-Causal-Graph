"""
Causal propagation module.

Responsible for propagating belief updates through the causal graph structure.
This ensures that changes in one belief (the antecedent) influence causally
downstream beliefs (the consequent).

Mechanism:
    For each causal link A -> B with weight w:
    delta_L(B) = rate * w * tanh(L(A) / 2)

    This means:
    - If A is strongly believed (L(A) > 0), B receives positive evidence.
    - If A is strongly disbelieved (L(A) < 0), B receives negative evidence.
    - If A is uncertain (L(A) ~ 0), B is unaffected.
    
    For negated antecedents (not_X -> Y), we use -L(X) to compute the effect.
"""

import math
from typing import Dict, List, Any, Optional

from core.data_structures import CharacterState, BeliefNode
from reasoning.belief_update import resolve_belief_conflicts


def _get_belief_log_odds(state: CharacterState, prop_name: str) -> Optional[float]:
    """
    Get the effective log-odds for a proposition, handling negations.
    
    For "not_X", returns -log_odds(X) if X exists.
    For "X", returns log_odds(X) directly.
    """
    prop_lower = prop_name.strip().lower()
    
    # Handle negated form: not_X means we want -log_odds(X)
    if prop_lower.startswith("not_"):
        base = prop_lower[4:]
        base_node = state.get_belief(base)
        if base_node is not None:
            return -base_node.log_odds
        node = state.get_belief(prop_lower)
        if node is not None:
            return node.log_odds
    elif prop_lower.startswith("~"):
        base = prop_lower[1:]
        base_node = state.get_belief(base)
        if base_node is not None:
            return -base_node.log_odds
        node = state.get_belief(prop_lower)
        if node is not None:
            return node.log_odds
    else:
        # Try direct lookup first
        node = state.get_belief(prop_lower)
        if node is not None:
            return node.log_odds

        # Maybe the negation exists
        neg_node = state.get_belief(f"not_{prop_lower}")
        if neg_node is not None:
            return -neg_node.log_odds
    
    return None


def _update_belief_log_odds(state: CharacterState, prop_name: str, delta: float) -> bool:
    """
    Update the log-odds for a proposition, handling negations.
    
    For "not_X", updates X's log_odds by -delta.
    For "X", updates X's log_odds by +delta.
    
    Returns True if update was applied.
    """
    prop_lower = prop_name.strip().lower()
    
    # Handle negated form: updating not_X by +delta means updating X by -delta
    if prop_lower.startswith("not_"):
        base = prop_lower[4:]
        base_node = state.get_belief(base)
        if base_node is not None:
            base_node.log_odds -= delta  # Note: subtract delta
            if abs(delta) > 0.01:
                base_node.add_evidence("causal_propagation")
            return True
        node = state.get_belief(prop_lower)
        if node is not None:
            node.log_odds += delta
            if abs(delta) > 0.01:
                node.add_evidence("causal_propagation")
            return True
    elif prop_lower.startswith("~"):
        base = prop_lower[1:]
        base_node = state.get_belief(base)
        if base_node is not None:
            base_node.log_odds -= delta
            if abs(delta) > 0.01:
                base_node.add_evidence("causal_propagation")
            return True
        node = state.get_belief(prop_lower)
        if node is not None:
            node.log_odds += delta
            if abs(delta) > 0.01:
                node.add_evidence("causal_propagation")
            return True
    else:
        # Try direct lookup first
        node = state.get_belief(prop_lower)
        if node is not None:
            node.log_odds += delta
            if abs(delta) > 0.01:
                node.add_evidence("causal_propagation")
            return True

        # Maybe the negation exists
        neg_node = state.get_belief(f"not_{prop_lower}")
        if neg_node is not None:
            neg_node.log_odds -= delta
            if abs(delta) > 0.01:
                neg_node.add_evidence("causal_propagation")
            return True
    
    return False


def propagate_causal_effects(
    state: CharacterState,
    propagation_rate: float = 0.1
) -> None:
    """
    Apply causal propagation rules to the belief network.

    Iterates through the character's causal_links and updates the log-odds
    of consequent beliefs based on the strength of antecedent beliefs.

    Mechanism:
        For each causal link A -> B with weight w:
        delta_L(B) = propagation_rate * w * tanh(L(A) / 2)
        
    Handles negated propositions (not_X) by using -log_odds(X).

    Preconditions
    -------------
    state : CharacterState
        Must contain initialized beliefs and causal_links.

    Procedure
    ---------
    1. For each causal link:
       a. Retrieve antecedent belief A (or its negation).
       b. Compute impact = propagation_rate * weight * tanh(L(A) / 2).
       c. Accumulate impacts for all consequents.
    2. Apply impacts to belief log-odds (handling negations).
    3. Resolve any belief conflicts after all updates.

    Postconditions
    --------------
    state.beliefs updated in place.

    Parameters
    ----------
    state : CharacterState
    propagation_rate : float
        Scalar to control the speed of propagation per turn.
    """
    if not state.causal_links:
        return

    updates: Dict[str, float] = {}

    # Iterate over all causal links
    for link in state.causal_links:
        ant_name = link["antecedent"]
        cons_name = link["consequent"]
        weight = link.get("weight", 1.0)

        # Use helper to get effective log-odds (handles not_X -> -log_odds(X))
        ant_log_odds = _get_belief_log_odds(state, ant_name)
        if ant_log_odds is None:
            continue

        # Compute impact using tanh(L(A)/2)
        # This reflects the strength of the belief A influencing B.
        strength = math.tanh(ant_log_odds / 2.0)
        impact = propagation_rate * weight * strength

        updates[cons_name] = updates.get(cons_name, 0.0) + impact

    # Apply accumulated updates using helper (handles negations)
    for prop, delta in updates.items():
        _update_belief_log_odds(state, prop, delta)

    # Ensure beliefs remain logically consistent
    resolve_belief_conflicts(state.beliefs)


def snapshot_belief_log_odds(state: CharacterState) -> Dict[str, float]:
    return {
        prop: node.log_odds
        for prop, node in state.beliefs.items()
    }

