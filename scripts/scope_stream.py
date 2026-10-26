"""Low-latency MJPEG reception, independent of the GUI and ROS."""


def iter_latest_jpegs(stream):
    """Yield the newest complete JPEG in each available network chunk.

    HTTPResponse.read(n) waits for n bytes on an unbounded MJPEG response.
    read1(n) returns currently available data after one buffered/socket read,
    so a small frame need not wait for subsequent frames to fill 64 KiB.
    """
    buf = b""
    while True:
        chunk = stream.read1(65536)
        if not chunk:
            return
        buf += chunk
        end = buf.rfind(b"\xff\xd9")
        if end != -1:
            start = buf.rfind(b"\xff\xd8", 0, end)
            if start != -1:
                yield buf[start:end + 2]
            buf = buf[end + 2:]
        if len(buf) > 4_000_000:
            buf = b""


class PlayoutBuffer:
    """Re-times irregularly arriving frames to a steady display cadence.

    The FCS only emits a frame when it finishes processing one, so arrivals are
    unevenly spaced (17-25 Hz, bursty). Showing each frame on arrival makes the
    picture stutter. Frames are queued and shown on a clock set by the smoothed
    arrival interval; the clock speeds up/slows down slightly with queue depth
    so the buffer neither drains (freeze) nor grows (latency).
    ``target_depth`` frames of delay are the price of the smoothing; 0 disables it.
    """

    def __init__(self, target_depth=2, max_depth=None, max_interval=0.2):
        self.target_depth = target_depth
        self.max_depth = max_depth if max_depth is not None else target_depth + 2
        self.max_interval = max_interval
        self.queue = []
        self.interval = 1 / 20  # smoothed arrival interval (s)
        self.next_play = 0.0
        self.primed = False
        self._last_arrival = None
        self.dropped = 0

    def push(self, item, now):
        if self._last_arrival is not None:
            gap = now - self._last_arrival
            if 0.0 < gap <= self.max_interval:  # long gaps are stalls, not cadence
                self.interval += 0.1 * (gap - self.interval)
        self._last_arrival = now
        self.queue.append(item)

    def pop(self, now):
        """Return the frame to show now, or None (keep showing the previous one)."""
        if self.target_depth <= 0:  # passthrough: newest frame, no delay
            if not self.queue:
                return None
            item = self.queue[-1]
            self.dropped += len(self.queue) - 1
            self.queue.clear()
            return item
        if not self.queue:
            return None
        if not self.primed:
            # wait until enough frames are buffered, but never longer than the
            # time they would take to arrive (a slow or stalled stream still shows)
            waited = now - self._last_arrival
            if len(self.queue) < self.target_depth and waited < 2.0 * self.interval:
                return None
            self.primed = True
            self.next_play = now
        if now < self.next_play:
            return None
        excess = len(self.queue) - self.max_depth
        if excess > 0:  # fell behind: drop the oldest to cap latency
            del self.queue[:excess]
            self.dropped += excess
        depth = len(self.queue)
        item = self.queue.pop(0)
        scale = min(1.15, max(0.85, 1.0 - 0.04 * (depth - self.target_depth)))
        self.next_play = max(self.next_play + self.interval * scale, now - self.interval)
        if not self.queue and now - self._last_arrival > 2.0 * self.interval:
            self.primed = False  # stream paused: rebuffer when it resumes
        return item

    def wait_s(self, now, cap=0.015):
        """How long the caller may sleep before the next frame could be due."""
        if self.target_depth <= 0 or not self.primed:
            return cap
        return min(cap, max(0.001, self.next_play - now))
