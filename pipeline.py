"""Runs hallway detection (depth) in its own thread, beside the sign loop.

Grounding DINO takes ~0.7 s a frame; the depth model a fraction of that. In one loop, depth
ran at DINO's pace (~1.4 frames/s live), which gives a side hallway, visible for 1-2 s with
the glasses, one or two looks. Here:

  frames ──FrameHub──┬─> depth consumer: DepthThread (depth -> scan -> tracker), at --depth-fps
                     └─> sign consumer:  the main loop (DINO, OCR, Gemma), at --fps

- live sources: each consumer takes the newest frame it hasn't seen (older ones are skipped),
  so neither waits for the other;
- files: each consumer gets every one of its frames in order (the reader waits for them), so
  results are the same as one loop and as intersection_replay.py; the run is just faster.

Side effects (speech, chat log, logs, video) stay in the main thread: it collects the depth
thread's results with DepthThread.drain()."""
import queue
import threading
import time

from direction import Heading

_END = object()


class _Consumer:
    def __init__(self, fps, keep_all):
        self.step = 1.0 / fps if fps else 0.0
        self.keep_all = keep_all
        self.last_t = -1e9
        self.q = queue.Queue(maxsize=2 if keep_all else 1)

    def wants(self, t):
        if t - self.last_t + 1e-6 >= self.step:
            self.last_t = t
            return True
        return False

    def put(self, item):
        if self.keep_all or item is _END:
            self.q.put(item)            # blocks: a file waits for its slowest consumer
            return
        try:                            # live: replace an unread frame with the newer one
            self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put(item)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is _END:
                return
            yield item


class FrameHub:
    """Reads `frames` (a generator at the higher of the two rates) in a thread and hands
    each consumer the frames at its own rate."""

    def __init__(self, frames, live, *rates):
        self.consumers = [_Consumer(r, keep_all=not live) for r in rates]
        self.error = None
        self._frames = frames
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for f in self._frames:
                if self._stop.is_set():
                    break
                for c in self.consumers:
                    if c.wants(f.t):
                        c.put(f)
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            for c in self.consumers:
                c.put(_END)

    def stop(self):
        """Stop reading (e.g. 'q' pressed) and unblock the consumers."""
        self._stop.set()
        for c in self.consumers:      # make room so the reader can finish
            try:
                while True:
                    c.q.get_nowait()
            except queue.Empty:
                pass


class DepthThread:
    """Runs an IntersectionWatcher over a consumer's frames. Results come back through
    drain() as (frame, heading, result, [(sentence or None, event)])."""

    def __init__(self, watcher, frames):
        self.watcher, self.frames = watcher, frames
        self.out = queue.Queue()
        self.heading = Heading()       # its own: the main thread updates the other one
        self.n, self.busy_s, self.error = 0, 0.0, None
        self._t_start = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for f in self.frames:
                if self._t_start is None:
                    self._t_start = time.time()
                t0 = time.time()
                self.heading.update(f.yaw_deg, f.t)
                res, said = self.watcher(f, self.heading)
                self.busy_s += time.time() - t0
                self.n += 1
                self.out.put((f, self.heading.value, res, said))
        except Exception as e:  # noqa: BLE001
            self.error = e
            raise

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.out.get_nowait())
            except queue.Empty:
                return items

    def join(self, timeout=None):
        self.thread.join(timeout)

    def rate(self):
        """(frames per second achieved, ms per frame of depth work)."""
        if not self.n or self._t_start is None:
            return 0.0, 0.0
        wall = max(time.time() - self._t_start, 1e-6)
        return self.n / wall, 1000 * self.busy_s / self.n
