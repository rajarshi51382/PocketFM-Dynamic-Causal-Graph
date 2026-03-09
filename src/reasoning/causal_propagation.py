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
"""

import math
from typing import Dict, List, Any

from core.data_structures import CharacterState, BeliefNode
from reasoning.belief_update import resolve_belief_conflicts

def propagate_causal_effects(
    state: CharacterState,
    propagation_rate: float = 0.1
) -> None:
    """
    Apply causal propagation rules to the belief network.

    Iterates through the character's causal_links and updates the log-odds
    of consequent beliefs based on the strength of antecedent beliefs.

    Preconditions
    -------------
    state : CharacterState
        Must contain initialized beliefs and causal_links.

    Procedure
    ---------
    1. For each link (antecedent, consequent, weight):
       a. Retrieve L(antecedent)
       b. Calculate influence: w * tanh(L(A)/2)
       c. Update L(consequent) += rate * influence

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

    # Accumulate updates first to ensure synchronous update (order-independent)
    updates: Dict[str, float] = {}

    for link in state.causal_links:
        ant_name = link["antecedent"]
        cons_name = link["consequent"]
        weight = link.get("weight", 1.0)

        ant_node = state.get_belief(ant_name)
        if not ant_node:
            continue
        
        # Calculate belief strength in [-1, 1]
        # tanh(x/2) is the standard mapping from log-odds to correlation coefficient
        strength = math.tanh(ant_node.log_odds / 2.0)

        # Calculate impact
        impact = propagation_rate * weight * strength
        
        updates[cons_name] = updates.get(cons_name, 0.0) + impact

    # Apply updates
    for prop, delta in updates.items():
        # Ensure the belief node exists; if not, create it (discovery/inference)
        node = state.get_belief(prop)
        if node is None:
            # Create new latent belief with neutral prior
            node = BeliefNode(proposition=prop, log_odds=0.0)
            state.add_belief(node)
            node.add_evidence("causal_inference")
        
        node.log_odds += delta
        # Mark source if significant update
        if abs(delta) > 0.1:
            node.add_evidence(f"inference_from_graph")

    # Ensure inferred beliefs are logically consistent
    resolve_belief_conflicts(state.beliefs)

