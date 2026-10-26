import http.server
import threading
import urllib.request

from scripts.scope_stream import iter_latest_jpegs


class Chunks:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def read1(self, size):
        return next(self.chunks, b"")


def test_fragmented_jpeg_and_multipart_headers():
    stream = Chunks([b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\xff",
                     b"\xd8image\xff", b"\xd9\r\n--frame\r\n", b"\xff\xd8next\xff\xd9"])
    assert list(iter_latest_jpegs(stream)) == [b"\xff\xd8image\xff\xd9", b"\xff\xd8next\xff\xd9"]


def test_backlog_keeps_latest_complete_frame_and_partial_next_frame():
    stream = Chunks([b"\xff\xd8old\xff\xd9\xff\xd8new\xff\xd9\xff\xd8par",
                     b"tial\xff\xd9"])
    assert list(iter_latest_jpegs(stream)) == [b"\xff\xd8new\xff\xd9", b"\xff\xd8partial\xff\xd9"]


def test_small_http_frame_is_delivered_without_waiting_for_another_frame():
    # Server withholds the next frame until the client consumes the first.
    # A read(65536) regression times out here; read1 delivers the 12 KiB frame.
    release = threading.Event()
    jpeg = b"\xff\xd8" + b"x" * 12000 + b"\xff\xd9"

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            self.wfile.flush()
            release.wait(2)

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/stream", timeout=0.5) as response:
            assert next(iter_latest_jpegs(response)) == jpeg
    finally:
        release.set()
        thread.join(timeout=3)
        server.server_close()


def _play(buf, arrivals, end, tick=0.002):
    """Feed frames at the given arrival times; return display times of each frame id."""
    shown, i, now = [], 0, 0.0
    while now <= end:
        while i < len(arrivals) and arrivals[i] <= now:
            buf.push(i, arrivals[i])
            i += 1
        item = buf.pop(now)
        if item is not None:
            shown.append((item, now))
        now += tick
    return shown


def test_playout_buffer_evens_out_bursty_arrivals():
    import random
    from scripts.scope_stream import PlayoutBuffer

    random.seed(1)
    arrivals, t = [], 0.0
    for _ in range(200):
        t += 0.05
        arrivals.append(t + random.uniform(-0.02, 0.02))  # 20 Hz with +-20 ms jitter
    arrivals.sort()

    def spread(shown):
        gaps = [b[1] - a[1] for a, b in zip(shown[20:], shown[21:])]
        return max(gaps) - min(gaps)

    raw = _play(PlayoutBuffer(target_depth=0), arrivals, 10.5)
    smooth = _play(PlayoutBuffer(target_depth=2), arrivals, 10.5)
    assert spread(smooth) < spread(raw) * 0.5
    assert [f for f, _ in smooth] == sorted(f for f, _ in smooth)  # never out of order


def test_playout_buffer_caps_latency_after_a_burst():
    from scripts.scope_stream import PlayoutBuffer

    buf = PlayoutBuffer(target_depth=2)
    for k in range(10):  # 10 frames land at once (e.g. after a hiccup)
        buf.push(k, 0.0)
    assert buf.pop(0.0) == 6  # oldest 6 dropped; backlog cannot exceed max_depth
    assert buf.dropped == 6


def test_playout_buffer_still_shows_a_slow_stream():
    from scripts.scope_stream import PlayoutBuffer

    buf = PlayoutBuffer(target_depth=2)
    buf.push("a", 0.0)
    assert buf.pop(0.0) is None  # waiting to fill
    assert buf.pop(0.2) == "a"  # but not forever
