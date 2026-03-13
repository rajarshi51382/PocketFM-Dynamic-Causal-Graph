"""
Tests for belief polarization behavior.

This tests the core requirement: when a user repeatedly sends messages
like "the castle is unsafe", beliefs should accumulate (polarize) over time.

The expected behavior:
1. First message: castle_is_safe decreases from initial value
2. Second message: castle_is_safe decreases further  
3. Nth message: castle_is_safe keeps decreasing (polarizing negative)
4. Causal propagation: king_is_wise also decreases as castle_is_safe decreases
"""

import pytest
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_structures import (
    BeliefNode,
    CharacterState,
    EventFrame,
    TraitState,
)
from extraction.event_extraction import extract_event, validate_event
from reasoning.belief_update import (
    apply_belief_updates,
    directional_alignment,
)
from reasoning.causal_propagation import (
    propagate_causal_effects,
    _get_belief_log_odds,
    _update_belief_log_odds,
)


def get_belief_log_odds(state: CharacterState, belief_name: str) -> float:
    """Helper to get belief log_odds with assertion that it exists."""
    belief = state.get_belief(belief_name)
    assert belief is not None, f"Belief {belief_name} should exist"
    return belief.log_odds


def set_belief_log_odds(state: CharacterState, belief_name: str, value: float) -> None:
    """Helper to set belief log_odds with assertion that it exists."""
    belief = state.get_belief(belief_name)
    assert belief is not None, f"Belief {belief_name} should exist"
    belief.log_odds = value


def create_test_character() -> CharacterState:
    """Create a character with default beliefs for testing."""
    traits = TraitState(traits={"bravery": 0.8, "trusting": 0.5})
    beliefs = {
        "castle_is_safe": BeliefNode("castle_is_safe", log_odds=1.5),
        "forest_is_dangerous": BeliefNode("forest_is_dangerous", log_odds=1.0),
        "king_is_wise": BeliefNode("king_is_wise", log_odds=0.5),
    }
    state = CharacterState(
        character_id="TestKnight",
        traits=traits,
        beliefs=beliefs,
    )
    state.add_causal_link(antecedent="castle_is_safe", consequent="king_is_wise", weight=0.8)
    state.add_causal_link(antecedent="not_castle_is_safe", consequent="not_king_is_wise", weight=0.8)
    return state


class TestEventExtractionForCastleUnsafe:
    """Test that event extraction correctly identifies castle safety statements."""
    
    def test_extract_castle_unsafe_direct(self):
        """'The castle is unsafe' should produce not_castle_is_safe."""
        event = extract_event("The castle is unsafe")
        event = validate_event(event, "The castle is unsafe", {"castle_is_safe", "king_is_wise"})
        
        # Should have the negated proposition
        assert "not_castle_is_safe" in event.propositions, f"Expected not_castle_is_safe, got {event.propositions}"
    
    def test_extract_castle_crumbling(self):
        """'Castle walls are crumbling' should produce not_castle_is_safe."""
        event = extract_event("I heard the castle walls are crumbling!")
        event = validate_event(event, "I heard the castle walls are crumbling!", {"castle_is_safe", "king_is_wise"})
        
        assert "not_castle_is_safe" in event.propositions, f"Expected not_castle_is_safe, got {event.propositions}"
    
    def test_extract_castle_dangerous(self):
        """'The castle is dangerous' should produce not_castle_is_safe."""
        event = extract_event("The castle is dangerous now")
        event = validate_event(event, "The castle is dangerous now", {"castle_is_safe", "king_is_wise"})
        
        assert "not_castle_is_safe" in event.propositions, f"Expected not_castle_is_safe, got {event.propositions}"
    
    def test_extract_castle_not_safe(self):
        """'The castle is not safe' should produce not_castle_is_safe."""
        event = extract_event("The castle is not safe anymore")
        event = validate_event(event, "The castle is not safe anymore", {"castle_is_safe", "king_is_wise"})
        
        assert "not_castle_is_safe" in event.propositions, f"Expected not_castle_is_safe, got {event.propositions}"
    
    def test_extract_castle_safe_positive(self):
        """'The castle is safe' should produce castle_is_safe (positive)."""
        event = extract_event("The castle is safe and secure")
        event = validate_event(event, "The castle is safe and secure", {"castle_is_safe", "king_is_wise"})
        
        assert "castle_is_safe" in event.propositions, f"Expected castle_is_safe, got {event.propositions}"
        assert "not_castle_is_safe" not in event.propositions


class TestDirectionalAlignmentForNegation:
    """Test directional_alignment correctly handles negated propositions."""
    
    def test_negated_prop_decreases_base_belief(self):
        """Event with not_castle_is_safe should return -1 for castle_is_safe belief."""
        event = EventFrame(propositions=["not_castle_is_safe"])
        belief = BeliefNode("castle_is_safe", log_odds=1.5)
        
        alignment = directional_alignment(event, belief)
        assert alignment == -1, f"Expected -1, got {alignment}"
    
    def test_positive_prop_increases_base_belief(self):
        """Event with castle_is_safe should return +1 for castle_is_safe belief."""
        event = EventFrame(propositions=["castle_is_safe"])
        belief = BeliefNode("castle_is_safe", log_odds=1.5)
        
        alignment = directional_alignment(event, belief)
        assert alignment == +1, f"Expected +1, got {alignment}"
    
    def test_unrelated_prop_returns_zero(self):
        """Event with unrelated prop should return 0."""
        event = EventFrame(propositions=["sky_is_blue"])
        belief = BeliefNode("castle_is_safe", log_odds=1.5)
        
        alignment = directional_alignment(event, belief)
        assert alignment == 0, f"Expected 0, got {alignment}"


class TestBeliefPolarization:
    """Test that repeated messages cause belief polarization."""
    
    def test_single_unsafe_message_decreases_belief(self):
        """One 'castle is unsafe' message should decrease castle_is_safe belief."""
        state = create_test_character()
        initial_log_odds = get_belief_log_odds(state, "castle_is_safe")
        
        event = EventFrame(propositions=["not_castle_is_safe"], confidence=1.0)
        apply_belief_updates(state, event, lambda_base=0.5)
        
        final_log_odds = get_belief_log_odds(state, "castle_is_safe")
        assert final_log_odds < initial_log_odds, \
            f"Expected decrease: {initial_log_odds} -> {final_log_odds}"

    def test_confidence_scales_update_magnitude(self):
        """Lower confidence should produce a smaller magnitude update."""
        state = create_test_character()
        initial = get_belief_log_odds(state, "castle_is_safe")

        low_conf_event = EventFrame(propositions=["not_castle_is_safe"], confidence=0.2)
        apply_belief_updates(state, low_conf_event, lambda_base=0.5)
        after_low = get_belief_log_odds(state, "castle_is_safe")

        # Reset and apply a high-confidence update
        set_belief_log_odds(state, "castle_is_safe", initial)
        high_conf_event = EventFrame(propositions=["not_castle_is_safe"], confidence=1.0)
        apply_belief_updates(state, high_conf_event, lambda_base=0.5)
        after_high = get_belief_log_odds(state, "castle_is_safe")

        assert (initial - after_high) > (initial - after_low), \
            "High-confidence update should have larger magnitude than low-confidence"
    
    def test_repeated_unsafe_messages_polarize_belief(self):
        """Repeated 'castle is unsafe' messages should keep decreasing belief."""
        state = create_test_character()
        log_odds_history = [get_belief_log_odds(state, "castle_is_safe")]
        
        # Send the same message 5 times
        for i in range(5):
            event = EventFrame(propositions=["not_castle_is_safe"], confidence=1.0)
            apply_belief_updates(state, event, lambda_base=0.5)
            log_odds_history.append(get_belief_log_odds(state, "castle_is_safe"))
        
        # Each subsequent message should decrease the log-odds further
        for i in range(1, len(log_odds_history)):
            assert log_odds_history[i] < log_odds_history[i-1], \
                f"Message {i}: expected decrease from {log_odds_history[i-1]} to {log_odds_history[i]}"
        
        # After 5 messages, belief should be significantly negative
        final = log_odds_history[-1]
        initial = log_odds_history[0]
        assert final < 0, f"After 5 negative messages, log_odds should be negative: {final}"
        assert final < initial - 2.0, f"Expected significant decrease: {initial} -> {final}"
    
    def test_end_to_end_castle_unsafe_extraction_and_update(self):
        """Full pipeline: extract event from message and update beliefs."""
        state = create_test_character()
        initial_castle = get_belief_log_odds(state, "castle_is_safe")
        
        # Simulate actual user message
        user_message = "The castle is unsafe!"
        event = extract_event(user_message)
        event = validate_event(event, user_message, state.belief_schema)
        
        # Apply belief update
        apply_belief_updates(state, event, lambda_base=0.5)
        
        final_castle = get_belief_log_odds(state, "castle_is_safe")
        
        # Belief should decrease
        assert final_castle < initial_castle, \
            f"Expected castle_is_safe to decrease: {initial_castle} -> {final_castle}"


class TestCausalPropagationWithNegation:
    """Test causal propagation handles negated beliefs correctly."""
    
    def test_get_belief_log_odds_for_negation(self):
        """_get_belief_log_odds should return negative for not_X when X exists."""
        state = create_test_character()
        
        # Direct lookup
        direct = _get_belief_log_odds(state, "castle_is_safe")
        assert direct == 1.5, f"Expected 1.5, got {direct}"
        
        # Negated lookup (should return -log_odds)
        negated = _get_belief_log_odds(state, "not_castle_is_safe")
        assert negated == -1.5, f"Expected -1.5, got {negated}"
    
    def test_update_belief_log_odds_for_negation(self):
        """_update_belief_log_odds should handle not_X by updating X inversely."""
        state = create_test_character()
        initial = get_belief_log_odds(state, "castle_is_safe")
        
        # Update the negation
        _update_belief_log_odds(state, "not_castle_is_safe", 0.5)
        
        # This should DECREASE castle_is_safe (since we increased not_castle_is_safe)
        final = get_belief_log_odds(state, "castle_is_safe")
        assert final == initial - 0.5, f"Expected {initial - 0.5}, got {final}"
    
    def test_causal_propagation_cascades_to_downstream(self):
        """When castle_is_safe decreases, king_is_wise should also decrease via causal link."""
        state = create_test_character()
        initial_castle = get_belief_log_odds(state, "castle_is_safe")
        initial_king = get_belief_log_odds(state, "king_is_wise")
        
        # Decrease castle_is_safe significantly
        set_belief_log_odds(state, "castle_is_safe", -1.0)
        
        # Propagate causal effects
        propagate_causal_effects(state, propagation_rate=0.5)
        
        final_king = get_belief_log_odds(state, "king_is_wise")
        
        # king_is_wise should decrease because castle_is_safe is now negative
        # The link castle_is_safe -> king_is_wise with negative L(castle) should push king negative
        assert final_king < initial_king, \
            f"Expected king_is_wise to decrease: {initial_king} -> {final_king}"
    
    def test_repeated_messages_with_causal_propagation(self):
        """Full simulation: repeated 'castle unsafe' with causal propagation."""
        state = create_test_character()
        
        castle_history = [get_belief_log_odds(state, "castle_is_safe")]
        king_history = [get_belief_log_odds(state, "king_is_wise")]
        
        # Send 7 messages to ensure the antecedent becomes strongly negative
        for i in range(7):
            event = EventFrame(propositions=["not_castle_is_safe"], confidence=1.0)
            apply_belief_updates(state, event, lambda_base=0.5)
            propagate_causal_effects(state, propagation_rate=0.2)

            castle_history.append(get_belief_log_odds(state, "castle_is_safe"))
            king_history.append(get_belief_log_odds(state, "king_is_wise"))

        # Castle should monotonically decrease
        for i in range(1, len(castle_history)):
            assert castle_history[i] < castle_history[i-1], \
                f"Castle: expected decrease at step {i}"

        # King should also trend downward (though may not be strictly monotonic
        # due to the tanh function - it depends on the sign of castle)
        assert king_history[-1] < king_history[0], \
            f"King should decrease overall: {king_history[0]} -> {king_history[-1]}"


class TestIntegrationWithPresets:
    """Test with the actual preset messages from the Streamlit app."""
    
    def test_castle_unsafe_preset(self):
        """Test with the exact preset message."""
        state = create_test_character()
        initial = get_belief_log_odds(state, "castle_is_safe")
        
        message = "I heard the castle walls are crumbling and it's no longer safe!"
        event = extract_event(message)
        event = validate_event(event, message, state.belief_schema)
        
        apply_belief_updates(state, event, lambda_base=0.5)
        
        final = get_belief_log_odds(state, "castle_is_safe")
        assert final < initial, f"Expected decrease: {initial} -> {final}"
    
    def test_king_betrayal_preset(self):
        """Test king betrayal message."""
        state = create_test_character()
        initial = get_belief_log_odds(state, "king_is_wise")
        
        message = "The king has betrayed the entire kingdom — he's a liar!"
        event = extract_event(message)
        event = validate_event(event, message, state.belief_schema)
        
        apply_belief_updates(state, event, lambda_base=0.5)
        
        final = get_belief_log_odds(state, "king_is_wise")
        assert final < initial, f"Expected decrease: {initial} -> {final}"
    
    def test_forest_safe_preset(self):
        """Test forest safety message."""
        state = create_test_character()
        initial = get_belief_log_odds(state, "forest_is_dangerous")
        
        message = "Actually, the forest has been cleared; it's perfectly safe now."
        event = extract_event(message)
        event = validate_event(event, message, state.belief_schema)
        
        apply_belief_updates(state, event, lambda_base=0.5)
        
        final = get_belief_log_odds(state, "forest_is_dangerous")
        # This should DECREASE forest_is_dangerous (saying forest is safe)
        assert final < initial, f"Expected decrease: {initial} -> {final}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
