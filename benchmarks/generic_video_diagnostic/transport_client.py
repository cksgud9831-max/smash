"""Windows GET/MJPEG comparison; no application imports or display window."""
import json
import time
import urllib.request
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'results' / 'generic_video_diagnostic' / 'transport_comparison_valid'
OUT.mkdir(parents=True, exist_ok=True)
base = 'http://127.0.0.1:18766'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def get(path):
    return opener.open(base + path, timeout=5)

def part(response):
    line = response.readline()
    while line in (b'\r\n', b'\n'):
        line = response.readline()
    if line.strip() != b'--frame':
        raise RuntimeError(f'Bad multipart boundary {line!r}')
    headers = {}
    while True:
        line = response.readline()
        if line in (b'\r\n', b'\n'):
            break
        if not line:
            raise EOFError('Stream ended in headers')
        k, v = line.decode().split(':', 1)
        headers[k.lower()] = v.strip()
    length = int(headers['content-length'])
    data = response.read(length)
    if len(data) != length:
        raise EOFError('Incomplete JPEG')
    return headers, data

def stats(values):
    return {'median': float(np.median(values)), 'p95': float(np.percentile(values,95)), 'mean': float(np.mean(values))}

summaries = []
for repeat in range(3):
    cases = [('640x480','get'),('640x480','stream'),('1280x720','get'),('1280x720','stream')]
    if repeat % 2:
        cases.reverse()
    for size, mode in cases:
        name = f'{repeat+1}_{size}_{mode}'
        with get('/configure/'+size) as r:
            r.read()
        response = get('/stream') if mode == 'stream' else None
        previous = -1
        rows = []
        try:
            for i in range(210):
                start = time.perf_counter_ns()
                if response is None:
                    with get('/frame?after='+str(previous)) as r:
                        data = r.read()
                        headers = {k.lower():v for k,v in r.headers.items()}
                else:
                    headers,data = part(response)
                received = time.perf_counter_ns()
                image = cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                done = time.perf_counter_ns()
                if image is None:
                    raise RuntimeError('JPEG decode failure')
                row = {'sequence':int(headers['x-sequence']), 'generated_ns':int(headers['x-generated-ns']),
                       'encoded_ns':int(headers['x-encoded-ns']), 'received_ns':received,
                       'read_call_ms':(received-start)/1e6, 'decode_ms':(done-received)/1e6, 'bytes':len(data)}
                previous = row['sequence']
                if i >= 30:
                    rows.append(row)
        finally:
            if response is not None:
                response.close()
        for row in rows:
            row['encode_ms'] = (row['encoded_ns']-row['generated_ns'])/1e6
        sequences = [r['sequence'] for r in rows]
        gaps = np.diff([r['received_ns'] for r in rows])/1e6
        duration = (rows[-1]['received_ns']-rows[0]['received_ns'])/1e9
        summary = {'case':name, 'frames':len(rows), 'receive_hz':(len(rows)-1)/duration,
                   'unique_sequences':len(set(sequences)), 'duplicates':len(rows)-len(set(sequences)),
                   'skipped_sequences':sum(max(0,b-a-1) for a,b in zip(sequences,sequences[1:])),
                   'arrival_gap_ms':stats(gaps), 'gaps_over_50ms':int(sum(gaps>50)),
                   'decode_ms':stats([r['decode_ms'] for r in rows]),
                   'encode_ms':stats([r['encode_ms'] for r in rows])}
        summaries.append(summary)
        (OUT/(name+'.json')).write_text(json.dumps({'summary':summary,'samples':rows},indent=2))
        (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
        print(json.dumps(summary),flush=True)
