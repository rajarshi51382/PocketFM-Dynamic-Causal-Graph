
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from core.data_structures import CharacterState, WorldState, BeliefNode, RelationshipState, EventFrame
from simulation.simulation_loop import simulation_turn
from reasoning.causal_propagation import propagate_causal_effects
from reasoning.belief_update import DIRECT_OBSERVATION

def test_perception_integration():
    print("Testing Perception Integration...")
    world = WorldState()
    # Add a visible object
    world.update_object("treasure_chest", "open")
    
    char = CharacterState(character_id="hero")
    
    # Run a turn with empty message
    simulation_turn("hello", char, world)
    
    # Hero should now have a belief about the chest
    belief = char.get_belief("treasure_chest_is_open")
    if belief and belief.probability > 0.6:
        print("SUCCESS: Character perceived the world state.")
    else:
        print(f"FAILURE: Character did not perceive the world. Belief: {belief}")

def test_causal_conflict_resolution():
    print("\nTesting Causal Conflict Resolution...")
    char = CharacterState(character_id="hero")
    # P: castle_is_safe, Q: king_is_wise. Link: P -> Q (weight 1.0)
    char.add_belief(BeliefNode("castle_is_safe", log_odds=2.0))
    char.add_belief(BeliefNode("not_king_is_wise", log_odds=2.0)) # Strongly believes NOT king_is_wise
    char.add_causal_link("castle_is_safe", "king_is_wise", weight=10.0) # Very strong link
    
    # Propagate
    propagate_causal_effects(char, propagation_rate=1.0)
    
    # Check if king_is_wise and not_king_is_wise sum to 1.0
    b_pos = char.get_belief("king_is_wise")
    b_neg = char.get_belief("not_king_is_wise")
    
    if b_pos and b_neg:
        p_sum = b_pos.probability + b_neg.probability
        print(f"Probabilities: {b_pos.probability:.3f} + {b_neg.probability:.3f} = {p_sum:.3f}")
        if abs(p_sum - 1.0) < 1e-5:
            print("SUCCESS: Causal propagation resolved conflicts.")
        else:
            print("FAILURE: Causal propagation did not resolve conflicts.")
    else:
        print(f"FAILURE: Beliefs missing. pos: {b_pos}, neg: {b_neg}")

def test_knowledge_boundary():
    print("\nTesting Knowledge Boundary...")
    world = WorldState(timeline_index=10)
    char = CharacterState(character_id="hero", knowledge_boundary=5)
    
    # User tries to tell hero something at t=10
    simulation_turn("The dragon is dead", char, world)
    
    belief = char.get_belief("dragon_is_dead")
    if not belief or belief.log_odds == 0:
        print("SUCCESS: Knowledge boundary blocked the update.")
    else:
        print(f"FAILURE: Knowledge boundary was ignored. Log-odds: {belief.log_odds}")

if __name__ == "__main__":
    test_perception_integration()
    test_causal_conflict_resolution()
    test_knowledge_boundary()
