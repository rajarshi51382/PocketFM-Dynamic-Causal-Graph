import pytest

import json

from state.timeline_seeds import TIMELINE_SEEDS, create_character_state_for_seed, load_timeline_seeds


def test_seed_keys_are_defined():
    assert "baseline" in TIMELINE_SEEDS
    assert "after_betrayal" in TIMELINE_SEEDS
    assert "after_peace" in TIMELINE_SEEDS


def test_seed_populates_beliefs_and_relationships():
    state = create_character_state_for_seed("after_betrayal")
    assert state.timeline_index == TIMELINE_SEEDS["after_betrayal"]["timeline_index"]
    belief = state.get_belief("king_is_wise")
    assert belief is not None
    assert belief.log_odds == pytest.approx(-0.6)
    assert "king" in state.relationships
    assert state.relationships["king"].trust == pytest.approx(0.2)


def test_unknown_seed_raises():
    with pytest.raises(ValueError):
        create_character_state_for_seed("missing_seed")


def test_load_timeline_seeds_from_file(tmp_path):
    payload = {
        "custom": {
            "label": "Custom",
            "timeline_index": 2,
            "beliefs": {"castle_is_safe": 0.4},
            "relationships": {},
        }
    }
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    seeds = load_timeline_seeds(str(path))
    assert "custom" in seeds
    state = create_character_state_for_seed("custom", seeds=seeds)
    belief = state.get_belief("castle_is_safe")
    assert belief is not None
    assert belief.log_odds == pytest.approx(0.4)
