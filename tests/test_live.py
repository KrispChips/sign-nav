"""The live link (live.py, main.py --speak-to/--imu-from) against tools/fake_phone.py."""
import json
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fake_phone  # noqa: E402
from live import PhoneSpeaker, TcpImu, parse_addr  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


class Collect(socketserver.StreamRequestHandler):
    lines, conns = [], []

    def handle(self):
        Collect.conns.append(self.request)
        try:
            for raw in self.rfile:
                Collect.lines.append(json.loads(raw))
        except (OSError, ValueError):
            pass


def stop_server(srv):
    """Like the app stopping: the listener and its open connections close."""
    srv.shutdown()
    srv.server_close()
    for c in Collect.conns:
        try:
            c.shutdown(socket.SHUT_RDWR)
            c.close()
        except OSError:
            pass
    Collect.conns = []


def test_parse_addr():
    assert parse_addr("s26", 9002) == ("s26", 9002)
    assert parse_addr("100.64.1.2:9100", 9002) == ("100.64.1.2", 9100)


def test_speaker_waits_for_phone_and_drops_expired():
    port = free_port()
    sp = PhoneSpeaker(("127.0.0.1", port), ttl={"place": 0.3})
    sp.send("Hallway ahead on your left.", "place")    # phone not up yet
    sp.send("EXIT, to the right.", "sign")
    time.sleep(0.6)                                    # the place warning expires meanwhile
    Collect.lines = []
    srv = fake_phone._TCP(("127.0.0.1", port), Collect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert wait_for(lambda: Collect.lines)
        time.sleep(0.2)
        assert [m["text"] for m in Collect.lines] == ["EXIT, to the right."]
        m = Collect.lines[0]
        assert m["kind"] == "sign" and m["ttl_s"] == 10.0 and abs(m["t"] - time.time()) < 5
    finally:
        sp.close()
        srv.shutdown()
        srv.server_close()


def test_speaker_reconnects_after_phone_restart():
    port = free_port()
    Collect.lines = []
    srv = fake_phone._TCP(("127.0.0.1", port), Collect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    sp = PhoneSpeaker(("127.0.0.1", port))
    try:
        sp.send("one")
        assert wait_for(lambda: len(Collect.lines) == 1)
        stop_server(srv)                                # the app stops
        assert wait_for(lambda: not sp.connected.is_set())
        srv = fake_phone._TCP(("127.0.0.1", port), Collect)   # ... and comes back
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        assert wait_for(lambda: sp.connected.is_set(), 15)
        sp.send("two")
        assert wait_for(lambda: [m["text"] for m in Collect.lines] == ["one", "two"])
    finally:
        sp.close()
        stop_server(srv)


def test_tcp_imu_reads_yaw():
    port = free_port()
    fake_phone.YawHandler.sweep = 30.0
    srv = fake_phone._TCP(("127.0.0.1", port), fake_phone.YawHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    imu = TcpImu(("127.0.0.1", port), gyro_sign=-1.0)
    try:
        assert wait_for(lambda: imu.n > 20)
        y = imu.yaw_at(0.0)
        assert y is not None and -30.5 <= y <= 30.5
    finally:
        imu.close()
        srv.shutdown()
        srv.server_close()
    assert wait_for(lambda: imu.yaw_at(0.0) is None, 3)    # stale once the phone is gone


@pytest.fixture
def corridor_video(tmp_path):
    import cv2
    import numpy as np
    import corridor_world as cw
    wd, _ = cw.scene_cross()
    path = tmp_path / "walk.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (216, 288))
    for t, y, inv, K in cw.walk(wd, 12, 22, fps=10, size=(216, 288), K=cw.camera_K(216, 288, 70)):
        vw.write(cv2.cvtColor(np.clip(inv * 60, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    vw.release()
    return path


def test_main_live_against_fake_phone(monkeypatch, tmp_path, corridor_video, capsys):
    """main.py on the fake phone's MJPEG stream speaks the crossing through --speak-to."""
    from test_pipeline import _stub_models
    _stub_models(monkeypatch)
    ports = [free_port() for _ in range(3)]
    fake_phone.SpeechHandler.spoken = []
    servers = fake_phone.serve(str(corridor_video), host="127.0.0.1", http_port=ports[0],
                               speech_port=ports[1], imu_port=ports[2], fps=10, loop=False)
    import main as main_mod      # (time.sleep left alone: the fake phone plays in real time)
    monkeypatch.setattr(sys, "argv", [
        "main.py", "--source", f"http://127.0.0.1:{ports[0]}/frames.mjpg", "--no-reconnect",
        "--speak-to", f"127.0.0.1:{ports[1]}", "--imu-from", f"127.0.0.1:{ports[2]}",
        "--fps", "10", "--hfov", "70", "--no-filter", "--no-legibility",
        "--debug-dir", str(tmp_path / "debug")])
    try:
        main_mod.main()
        printed = capsys.readouterr().out
        assert "[source] connected: 216x288" in printed
        assert wait_for(lambda: any("Hallway" in t for _, t, _ in fake_phone.SpeechHandler.spoken))
        kinds = {k for k, t, _ in fake_phone.SpeechHandler.spoken if "Hallway" in t}
        assert kinds == {"place"}
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()
