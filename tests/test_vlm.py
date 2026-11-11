"""Prompt parsing and the arrow-evaluation scoring (no model calls)."""
import json

from vlm import PROMPTS, parse
from vlm_eval import arrow_close, score_sign, summarize


def test_all_prompts_exist():
    assert set(PROMPTS) == {"v1", "v2", "v3", "v4"}
    assert "in English" in PROMPTS["v4"] and "Never describe the" in PROMPTS["v4"]
    assert PROMPTS["v4"].split("- Each destination once")[0] == PROMPTS["v2"].split("- Copy text exactly")[0]
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


def test_parse_drops_entries_with_no_latin_text():
    raw = json.dumps({"entries": [{"text": "\u80be\u810f\u90e8", "arrow": "left"},
                                  {"text": "\u80be\u810f\u90e8 Kidney Unit", "arrow": "left"},
                                  {"text": "\u2191", "arrow": "up"}]})
    assert [e["text"] for e in parse(raw)] == ["\u80be\u810f\u90e8 Kidney Unit"]


def test_named_pictograms_are_not_invented():
    gt = [("EXIT", "left")]
    read = [{"text": "Exit", "arrow": "left"}, {"text": "Toilets", "arrow": "right"},
            {"text": "Wheelchair accessibility", "arrow": "right"}, {"text": "Steven Medlock Street", "arrow": "up"}]
    s = score_sign(gt, read, symbols=["TOILET", "WHEELCHAIR ACCESSIBILITY"])
    assert s["invented"] == 1 and s["symbols_found"] == [True, True]


def test_default_prompt_is_v4():
    import vlm
    assert vlm.DEFAULT_PROMPT == "v4"


def test_reader_retries_once(monkeypatch):
    import vlm
    calls = []

    class Client:
        def chat(self, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("llama-server process has terminated")
            return {"message": {"content": json.dumps({"entries": [{"text": "Exit", "arrow": "left"}]})}}
    r = vlm.SignReader.__new__(vlm.SignReader)
    r.client, r.model, r.version, r.prompt = Client(), "m", "v4", vlm.PROMPTS["v4"]
    r.think, r.keep_alive = False, "2h"
    monkeypatch.setattr(vlm.time, "sleep", lambda s: None)
    import numpy as np
    entries, raw = r(np.zeros((20, 40, 3), np.uint8))
    assert len(calls) == 2 and entries == [{"text": "Exit", "arrow": "left"}]
