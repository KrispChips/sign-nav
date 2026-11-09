"""The depth thread and frame hub (pipeline.py), and main.py's loop with them."""
import json
import sys
import time
import types

import cv2
import numpy as np
import pytest

import corridor_world as cw
from intersection import IntersectionWatcher
from pipeline import DepthThread, FrameHub
from sources import Frame


def frames(n, fps=10.0, delay=0.0):
    for i in range(n):
        if delay:
            time.sleep(delay)
        yield Frame(np.zeros((4, 4, 3), np.uint8), i / fps, np.eye(3), None)


def test_file_mode_gives_each_consumer_every_frame_at_its_rate():
    hub = FrameHub(frames(40), False, 5.0, 10.0)
    slow, fast = [], []

    def eat(c, out, pause):
        for f in c:
            out.append(round(f.t, 2))
            time.sleep(pause)
    import threading
    th = threading.Thread(target=eat, args=(hub.consumers[1], fast, 0.0))
    th.start()
    eat(hub.consumers[0], slow, 0.01)        # the slow sign loop doesn't make depth skip
    th.join()
    assert fast == [round(i / 10, 2) for i in range(40)]
    assert slow == [round(i / 10, 2) for i in range(0, 40, 2)]


def test_variable_frame_rate_keeps_every_pick():
    """The glasses record at a variable frame rate: the reader's 5 fps picks come
    0.19-0.23 s apart. Each must reach the consumers (a 'gap >= 0.2 s' rule dropped 1/3)."""
    from pipeline import _Consumer
    rng = np.random.default_rng(0)
    ts = np.cumsum(rng.uniform(0.028, 0.039, 3000))
    picks, next_t = [], 0.0
    for t in ts[ts < 89.4]:              # what sources.video_frames yields at 5 fps
        if t + 1e-6 < next_t:
            continue
        picks.append(t)
        next_t = max(next_t + 0.2, t)
    c = _Consumer(5.0, True)
    assert sum(c.wants(t) for t in picks) == len(picks)
    half = _Consumer(2.5, True)          # a slower consumer still gets ~every other one
    assert abs(sum(half.wants(t) for t in picks) - len(picks) / 2) <= 3


def test_live_mode_slow_consumer_gets_newest_frames_only():
    hub = FrameHub(frames(60, delay=0.005), True, 100.0, 100.0)
    slow, fast = [], []
    import threading
    th = threading.Thread(target=lambda: fast.extend(f.t for f in hub.consumers[1]))
    th.start()
    for f in hub.consumers[0]:
        slow.append(f.t)
        time.sleep(0.05)                      # a slow DINO
    th.join()
    assert len(fast) > 2 * len(slow)          # depth isn't held back by the slow loop
    assert slow == sorted(slow) and slow[-1] >= 5.0   # ... and the slow loop stays current


def test_depth_thread_same_events_as_inline():
    """Same frames, same events, whether depth runs inline or in its own thread."""
    from intersection import phrase
    wd, _ = cw.scene_cross()
    walk = list(cw.walk(wd, 5, 26))
    inline = IntersectionWatcher(None)
    want = [phrase(e) for t, y, inv, K in walk for e in inline.process(inv, K, (72, 96), t)[1]]

    maps = {round(t, 3): inv for t, y, inv, K in walk}
    threaded = IntersectionWatcher(None)

    def watch(frame, heading):          # the rendered map stands in for the depth model
        return threaded(frame, heading, inv=maps[round(frame.t, 3)])

    src = (Frame(np.zeros((96, 72, 3), np.uint8), t, K, None) for t, y, inv, K in walk)
    hub = FrameHub(src, False, 5.0, 5.0)
    dt = DepthThread(watch, iter(hub.consumers[1]))
    for _ in hub.consumers[0]:
        pass
    dt.join(10)
    got = [e["sentence"] for f, h, res, said in dt.drain() for s, e in said]
    assert got == want == ["Hallways cross ahead, on both sides."]
    assert dt.n == len(walk)


# ---- main.py end to end, with the heavy models stubbed out

@pytest.fixture
def synthetic_video(tmp_path):
    wd, _ = cw.scene_cross()
    path = tmp_path / "walk.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (216, 288))
    for t, y, inv, K in cw.walk(wd, 12, 22, fps=10, size=(216, 288), K=cw.camera_K(216, 288, 70)):
        vw.write(cv2.cvtColor(np.clip(inv * 60, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release()
    return path


def _stub_models(monkeypatch):
    import intersection

    class FakeDepth:            # the video's pixels are the rendered inverse depth
        def __init__(self, *a, **k):
            pass

        def __call__(self, img):
            time.sleep(0.005)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 60

    monkeypatch.setattr(intersection, "DepthEstimator", FakeDepth)
    monkeypatch.setattr(intersection, "resolve_model", lambda p=None: "fake.onnx")

    class FakeDetector:
        dropped, device = 0, "cpu"

        def __init__(self, **k):
            pass

        def __call__(self, img):
            time.sleep(0.02)    # slower than depth, like Grounding DINO
            return []
    monkeypatch.setitem(sys.modules, "detector",
                        types.SimpleNamespace(DISTRACTORS=(), SignDetector=FakeDetector))
    monkeypatch.setitem(sys.modules, "vlm", types.SimpleNamespace(SignReader=lambda *a, **k: None))


@pytest.mark.parametrize("extra", [[], ["--no-depth-thread"], ["--depth-fps", "10"]])
def test_main_announces_hallways(monkeypatch, tmp_path, synthetic_video, extra, capsys):
    _stub_models(monkeypatch)
    import main as main_mod
    monkeypatch.setattr(main_mod.time, "sleep", lambda s: None)   # skip the end-of-run waits
    out = tmp_path / "debug"
    monkeypatch.setattr(sys, "argv", ["main.py", "--source", str(synthetic_video), "--hfov", "70",
                                      "--no-tts", "--no-filter", "--no-legibility",
                                      "--debug-dir", str(out),
                                      "--out-video", str(tmp_path / "a.mp4")] + extra)
    main_mod.main()
    printed = capsys.readouterr().out
    spoken = [ln for ln in printed.splitlines() if ln.startswith("[SPEAK]")]
    # at a higher depth rate one side can be decided before the other is measured:
    # "Hallway ... right." + "Hallway on your left too." is the same crossing
    assert spoken in (["[SPEAK] Hallways cross ahead, on both sides."],
                      ["[SPEAK] Hallway ahead on your right.", "[SPEAK] Hallway on your left too."],
                      ["[SPEAK] Hallway ahead on your left.", "[SPEAK] Hallway on your right too."])
    lines = [json.loads(x) for x in (out / "intersections.jsonl").read_text().splitlines()]
    assert lines and all("t" in x for x in lines)
    if "--no-depth-thread" not in extra:
        assert "[depth] own thread" in printed and "[depth] " in printed
    assert (tmp_path / "a.mp4").exists()
