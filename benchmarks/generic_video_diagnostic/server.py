"""Standalone synthetic JPEG source; does not import application code."""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np

FRAMES = {}
for width, height in [(640, 480), (1280, 720)]:
    y, x = np.indices((height, width))
    frame = np.stack([(x % 256), (y % 256), ((x // 16 + y // 16) % 2 * 180 + 40)], axis=2).astype(np.uint8)
    FRAMES[f'/{width}x{height}'] = frame

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        frame = FRAMES.get(self.path)
        if frame is None:
            self.send_error(404)
            return
        start = time.perf_counter_ns()
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        encode_ms = (time.perf_counter_ns() - start) / 1e6
        if not ok:
            self.send_error(500)
            return
        data = encoded.tobytes()
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Encode-Ms', str(encode_ms))
        self.end_headers()
        self.wfile.write(data)

print(json.dumps({'ready': True, 'port': 18765, 'opencv': cv2.__version__}), flush=True)
ThreadingHTTPServer(('127.0.0.1', 18765), Handler).serve_forever()
