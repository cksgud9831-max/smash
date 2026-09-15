"""Windows receiver for a standalone WSL synthetic JPEG source."""
import json
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'results' / 'generic_video_diagnostic'
OUT.mkdir(parents=True, exist_ok=True)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
rows = []
raw_rows = []

def summary(values):
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'mean': float(np.mean(values))}

for repeat in range(2):
    cases = [('640x480', False), ('640x480', True), ('1280x720', False), ('1280x720', True)]
    if repeat:
        cases.reverse()
    for size, display in cases:
        name = f'{repeat+1}_{size}_{"display" if display else "no_display"}'
        samples = []
        if display:
            cv2.namedWindow('Generic video diagnostic', cv2.WINDOW_AUTOSIZE)
        for index in range(140):
            begin = time.perf_counter()
            with opener.open('http://127.0.0.1:18765/' + size, timeout=5) as response:
                data = response.read()
                encode_ms = float(response.headers['X-Encode-Ms'])
            received = time.perf_counter()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            decoded = time.perf_counter()
            if frame is None:
                raise RuntimeError('JPEG decode failed')
            if display:
                cv2.imshow('Generic video diagnostic', frame)
                cv2.waitKey(1)
            end = time.perf_counter()
            sample = {'http_ms': (received-begin)*1000, 'encode_ms': encode_ms,
                      'http_minus_encode_ms': (received-begin)*1000-encode_ms,
                      'decode_ms': (decoded-received)*1000,
                      'display_call_ms': (end-decoded)*1000,
                      'total_work_ms': (end-begin)*1000, 'jpeg_bytes': len(data)}
            if index >= 20:
                samples.append(sample)
            time.sleep(max(0, 1/30 - (time.perf_counter()-begin)))
        if display:
            cv2.destroyAllWindows()
        row = {'case': name, 'frames': len(samples), 'over_33ms': sum(s['total_work_ms'] > 1000/30 for s in samples),
               'metrics': {key: summary([s[key] for s in samples]) for key in samples[0]}}
        rows.append(row)
        raw_rows.append({'case': name, 'samples': samples})
        (OUT/'windows_summary.json').write_text(json.dumps(rows, indent=2))
        (OUT/'windows_samples.json').write_text(json.dumps(raw_rows, indent=2))
        print(json.dumps(row), flush=True)
