from core.data_structures import CharacterState, BeliefNode
from reasoning.verifier import verify_dialogue


def test_verifier_allows_consistent_response():
    state = CharacterState()
    state.add_belief(BeliefNode("king_is_wise", log_odds=3.0))

    response = "The king is wise."

    is_valid, violations = verify_dialogue(response, state)

    assert is_valid is True
    assert violations == []


def test_verifier_flags_contradiction_for_is_proposition():
    state = CharacterState()
    state.add_belief(BeliefNode("king_is_wise", log_odds=3.0))

    response = "The king is not wise."

    is_valid, violations = verify_dialogue(response, state)

    assert is_valid is False
    assert "contradicts_belief:king_is_wise" in violations