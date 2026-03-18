import os

from extraction import event_extraction


def test_embedding_fallback_maps_ood_phrase(monkeypatch):
    def fake_embedding(text: str) -> list[float]:
        text_lower = text.lower()
        if "castle" in text_lower:
            return [1.0, 0.0]
        if "king" in text_lower:
            return [0.0, 1.0]
        return [0.5, 0.0]

    monkeypatch.setattr(event_extraction.llm_client, "configure_client", lambda: False)
    monkeypatch.setattr(event_extraction.llm_client, "is_embedding_available", lambda: True)
    monkeypatch.setattr(event_extraction.llm_client, "get_embedding", fake_embedding)

    event = event_extraction.extract_event("The castle feels safer now.")
    assert "castle_is_safe" in event.propositions


def test_embedding_fallback_skips_when_unavailable(monkeypatch):
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setattr(event_extraction.llm_client, "configure_client", lambda: False)
    monkeypatch.setattr(event_extraction.llm_client, "is_embedding_available", lambda: False)

    event = event_extraction.extract_event("The castle feels safer now.")
    assert event.propositions
