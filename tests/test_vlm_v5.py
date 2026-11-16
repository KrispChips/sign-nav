"""Prompt v5: escalator / stairs ways and the "from" field survive parsing and scoring."""
import json

from eval_signgapore import to_sg
from vlm import PROMPTS, parse


def _raw(*entries):
    return json.dumps({"entries": list(entries)})


def test_v5_registered():
    assert "up-escalator" in PROMPTS["v5"]


def test_escalator_way_becomes_arrow_plus_via():
    out = parse(_raw({"text": "Dining @ B3", "from": "text", "arrow": "down-escalator"},
                     {"text": "Taxi", "from": "symbol", "arrow": "right"},
                     {"text": "MRT", "arrow": "up escalator"},
                     {"text": "Car park", "arrow": "up-stairs"}))
    assert out == [
        {"text": "Dining @ B3", "arrow": "down", "via": "escalator", "from": "text"},
        {"text": "Taxi", "arrow": "right", "from": "symbol"},
        {"text": "MRT", "arrow": "up", "via": "escalator"},
        {"text": "Car park", "arrow": "up", "via": "stairs"},
    ]


def test_old_answers_unchanged():
    assert parse(_raw({"text": "B2", "arrow": "→"})) == [{"text": "B2", "arrow": "right"}]


def test_to_sg_uses_via():
    assert to_sg("Dining @ B3", "down", "escalator") == "down-escalator"
    assert to_sg("Taxi", "up", "stairs") == "up-stairs"
    assert to_sg("Taxi", "up") == "forward"
    assert to_sg("Up escalator", "up") == "up-escalator"   # old text-based rule still applies
