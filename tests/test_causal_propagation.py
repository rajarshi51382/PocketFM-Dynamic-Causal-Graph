
import math
import pytest
from src.core.data_structures import CharacterState, BeliefNode
from src.reasoning.causal_propagation import propagate_causal_effects

def test_add_causal_link_ensures_nodes():
    state = CharacterState()
    
    # Add a base belief
    state.add_belief(BeliefNode("A", log_odds=2.0))
    
    # Add a causal link with a negated antecedent and a new consequent
    state.add_causal_link(antecedent="not_A", consequent="B", weight=0.8)
    
    # Check if not_A was created with negated log-odds
    assert "not_a" in state.beliefs
    assert math.isclose(state.beliefs["not_a"].log_odds, -2.0)
    
    # Check if B was created with neutral log-odds
    assert "b" in state.beliefs
    assert state.beliefs["b"].log_odds == 0.0
    
    # Propagate
    propagate_causal_effects(state, propagation_rate=1.0)
    
    # Check if B was updated
    # strength = tanh(-2/2) = -0.76159
    # delta = 1.0 * 0.8 * -0.76159 = -0.60927
    assert math.isclose(state.beliefs["b"].log_odds, -0.60927, rel_tol=1e-4)

def test_add_causal_link_consistency_with_existing_negation():
    state = CharacterState()
    
    # Add a negated belief first
    state.add_belief(BeliefNode("not_A", log_odds=-3.0))
    
    # Add a causal link using the positive form
    state.add_causal_link(antecedent="A", consequent="B", weight=1.0)
    
    # Check if A was created with negated log-odds of not_A
    assert "a" in state.beliefs
    assert state.beliefs["a"].log_odds == 3.0

def test_get_parents_and_children():
    state = CharacterState()
    state.add_causal_link("A", "B", 1.0)
    state.add_causal_link("A", "C", 0.5)
    state.add_causal_link("D", "B", 0.2)
    
    children_a = state.get_children("A")
    assert len(children_a) == 2
    assert any(l["consequent"] == "B" for l in children_a)
    assert any(l["consequent"] == "C" for l in children_a)
    
    parents_b = state.get_parents("B")
    assert len(parents_b) == 2
    assert any(l["antecedent"] == "A" for l in parents_b)
    assert any(l["antecedent"] == "D" for l in parents_b)
