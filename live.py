"""PC side of the live link to the phone app (android/). Protocol: docs/LIVE.md.

The phone is the server for everything, so only its address is needed (its Tailscale name
or IP, or localhost with `adb forward`) and the PC needs no inbound firewall rules:

  http://<phone>:8080/frames.mjpg   glasses frames, MJPEG     -> main.py --source
  tcp  <phone>:9002                 sentences to speak        <- PhoneSpeaker (--speak-to)
  tcp  <phone>:9003                 head yaw from the glasses -> TcpImu (--imu-from)

Both TCP links reconnect on their own: the phone can come and go during a walk."""
import json
import queue
import socket
import threading
import time

TTL_S = {"place": 3.0, "sign": 10.0}   # a hallway warning said late is worse than none


def parse_addr(s, default_port):
    host, _, port = s.rpartition(":") if ":" in s else (s, "", "")
    return (host or s), int(port or default_port)


class _Reconnecting:
    """A TCP client that keeps reconnecting in a background thread."""

    def __init__(self, addr, name, retry_s=2.0):
        self.addr, self.name, self.retry_s = addr, name, retry_s
        self.sock = None
        self.connected = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        warned = False
        while not self._stop.is_set():
            try:
                s = socket.create_connection(self.addr, timeout=5)
                s.settimeout(None)
            except OSError as e:
                if not warned:
                    print(f"[{self.name}] waiting for {self.addr[0]}:{self.addr[1]} ({e})", flush=True)
                    warned = True
                time.sleep(self.retry_s)
                continue
            print(f"[{self.name}] connected to {self.addr[0]}:{self.addr[1]}", flush=True)
            warned = False
            self.sock = s
            self.connected.set()
            try:
                self.session(s)
            except OSError as e:
                print(f"[{self.name}] connection lost ({e}); reconnecting", flush=True)
            finally:
                self.connected.clear()
                self.sock = None
                try:
                    s.close()
                except OSError:
                    pass
            time.sleep(self.retry_s)

    def session(self, s):
        raise NotImplementedError

    def close(self):
        self._stop.set()
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass


class PhoneSpeaker(_Reconnecting):
    """Sends sentences to the phone app, which speaks them through the glasses.
    One JSON line each: {"text", "kind": "sign"|"place", "t": unix time, "ttl_s"}. The phone
    drops a sentence still waiting in its speech queue `ttl_s` after it arrived. Sentences
    decided while the phone is unreachable are held until they expire."""

    def __init__(self, addr, ttl=None):
        self.ttl = dict(TTL_S, **(ttl or {}))
        self.q = queue.Queue()
        super().__init__(addr, "speak-to")

    def send(self, text, kind="sign"):
        self.q.put({"text": text, "kind": kind, "t": round(time.time(), 3),
                    "ttl_s": self.ttl.get(kind, TTL_S["sign"])})

    def session(self, s):
        threading.Thread(target=self._watch_close, args=(s,), daemon=True).start()
        while True:
            msg = self.q.get()
            if msg is None:
                return
            if time.time() - msg["t"] > msg["ttl_s"]:
                print(f"[speak-to] expired before the phone was reachable: {msg['text']}", flush=True)
                continue
            try:
                s.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            except OSError:
                self.q.put(msg)           # resend after reconnecting, if still fresh
                raise

    def _watch_close(self, s):
        """The phone never sends anything back; a read returning means it hung up. Without
        this a dead link is only noticed at the next sentence."""
        try:
            while s.recv(1024):
                pass
        except OSError:
            pass
        if self.sock is s:
            self.q.put(None)              # wake the sender so it reconnects

    def pending(self):
        return self.q.qsize()


class TcpImu(_Reconnecting):
    """Head yaw from the glasses' motion sensors, streamed by the phone as JSON lines
    {"yaw_deg": float, ...} (degrees, increasing when turning right). Same interface as
    sources.UdpImu: yaw_at(t) gives the latest value, or None when there is none."""

    def __init__(self, addr, gyro_sign=1.0, stale_s=1.0):
        self.yaw, self.sign, self.stale_s = None, gyro_sign, stale_s
        self.last = 0.0
        self.n = 0
        super().__init__(addr, "imu-from")

    def session(self, s):
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                raise OSError("closed by the phone")
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                try:
                    msg = json.loads(line)
                    y = msg.get("yaw_deg")
                except (ValueError, AttributeError):
                    continue
                if y is not None:
                    self.yaw, self.last = self.sign * float(y), time.time()
                    self.n += 1

    def yaw_at(self, t):   # live: always the latest sample, if recent
        return self.yaw if time.time() - self.last <= self.stale_s else None
