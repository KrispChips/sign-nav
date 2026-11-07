"""Prompt parsing and the arrow-evaluation scoring (no model calls)."""
import json

from vlm import PROMPTS, parse
from vlm_eval import arrow_close, score_sign, summarize


def test_all_prompts_exist():
    assert set(PROMPTS) == {"v1", "v2", "v3"}
    assert "up-left" in PROMPTS["v2"] and '"arrows"' in PROMPTS["v3"]


def test_parse_v1_words_and_glyphs():
    raw = json.dumps({"entries": [{"text": "Exit", "arrow": "left"}, {"text": "Lifts", "arrow": "↗"},
                                  {"text": "Cafe", "arrow": "straight"}, {"text": "", "arrow": "up"}]})
    assert parse(raw) == [{"text": "Exit", "arrow": "left"}, {"text": "Lifts", "arrow": "up-right"},
                          {"text": "Cafe", "arrow": "up"}]


def test_parse_v3_arrow_ids():
    raw = json.dumps({"arrows": [{"id": "A", "direction": "up-left"}, {"id": "b", "direction": "right"}],
                      "entries": [{"text": "Lifts", "arrow": "A"}, {"text": "Exit", "arrow": "B"},
                                  {"text": "L2", "arrow": "none"}, {"text": "Cafe", "arrow": "down"}]})
    assert [e["arrow"] for e in parse(raw, "v3")] == ["up-left", "right", "none", "down"]


def test_parse_garbage():
    assert parse("not json") == [] and parse("[1, 2]") == [] and parse('{"entries": "x"}') == []


def test_arrow_close():
    assert arrow_close("up-left", "left") and arrow_close("up", "up-right") and arrow_close("left", "left")
    assert not arrow_close("left", "right") and not arrow_close("none", "up") and not arrow_close("up", "left")


def test_score_and_summary():
    gt = [("EXIT", "left"), ("LIFTS", "up-right"), ("L2", "none")]
    read = [{"text": "Exit", "arrow": "right"}, {"text": "Lifts", "arrow": "right"},
            {"text": "L2", "arrow": "none"}, {"text": "Surgical Ward", "arrow": "up"}]
    s = score_sign(gt, read)
    assert s["text_ok"] == [True, True, True] and s["invented"] == 1
    by = {a["gt"]: a for a in s["arrows"]}
    assert not by["left"]["exact"] and not by["left"]["close"]
    assert not by["up-right"]["exact"] and by["up-right"]["close"] and by["none"]["exact"]
    t = summarize([{"sign": "s", "view": 0, "score": s}, {"sign": "s", "view": 30, "score": s}])
    assert t["arrow_exact"][0] == 1 / 3 and t["stable_across_views"][0] == 1.0
    assert t["arrow_mixed_signs"][1] == 6
