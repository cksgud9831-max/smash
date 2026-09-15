"""Independent 30 Hz synthetic source, shared by HTTP GET and MJPEG tests."""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
import cv2
import numpy as np

frames = {}
for w, h in [(640, 480), (1280, 720)]:
    y, x = np.indices((h, w))
    frames[f'{w}x{h}'] = np.stack([x % 256, y % 256, (x // 16 + y // 16) % 2 * 180 + 40], axis=2).astype(np.uint8)
condition = threading.Condition()
size = '640x480'
latest = None

def produce():
    global latest
    deadline = time.monotonic()
    sequence = 0
    while True:
        time.sleep(max(0, deadline-time.monotonic()))
        with condition:
            selected = size
        start = time.monotonic_ns()
        ok, encoded = cv2.imencode('.jpg', frames[selected], [cv2.IMWRITE_JPEG_QUALITY, 80])
        encoded_at = time.monotonic_ns()
        if not ok:
            raise RuntimeError('Encoding failed')
        sequence += 1
        with condition:
            latest = (sequence, start, encoded_at, encoded.tobytes(), selected)
            condition.notify_all()
        deadline += 1/30
        if deadline < time.monotonic():
            deadline = time.monotonic()

class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *args):
        pass

    def send_bytes(self, data, mime='application/json', extra=None):
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        global size
        try:
            parsed = urlsplit(self.path)
            if self.path == '/clock':
                self.send_bytes(json.dumps({'ns': time.monotonic_ns()}).encode())
                return
            if self.path.startswith('/configure/'):
                selected = self.path.rsplit('/', 1)[-1]
                if selected not in frames:
                    self.send_error(400)
                    return
                with condition:
                    size = selected
                    condition.wait_for(lambda: latest is not None and latest[4] == size, timeout=2)
                self.send_bytes(b'{}')
                return
            if parsed.path not in ('/frame', '/stream'):
                self.send_error(404)
                return
            stream = parsed.path == '/stream'
            if stream:
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
            previous = int(parse_qs(parsed.query).get('after', ['-1'])[0])
            while True:
                with condition:
                    condition.wait_for(lambda: latest is not None and latest[0] != previous, timeout=2)
                    seq, generated, encoded_at, data, selected = latest
                previous = seq
                headers = {'X-Sequence': seq, 'X-Generated-Ns': generated, 'X-Encoded-Ns': encoded_at}
                if not stream:
                    self.send_bytes(data, 'image/jpeg', headers)
                    return
                part = '--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(data)) + '\r\n'
                part += ''.join(f'{k}: {v}\r\n' for k, v in headers.items()) + '\r\n'
                self.wfile.write(part.encode() + data + b'\r\n')
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

threading.Thread(target=produce, daemon=True).start()
print('READY generic transport server 18766', flush=True)
ThreadingHTTPServer(('127.0.0.1', 18766), Handler).serve_forever()
