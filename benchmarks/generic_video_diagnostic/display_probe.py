"""Independent Windows HighGUI timing diagnostic; no networking or app code."""
import json
import time
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'results' / 'generic_video_diagnostic' / 'display_comparison'
OUT.mkdir(parents=True, exist_ok=True)
WINDOW = 'Independent display diagnostic'
PERIOD_NS = 1_000_000_000 / 30

def stats(values):
    return {'median':float(np.median(values)), 'p95':float(np.percentile(values,95)), 'mean':float(np.mean(values))}

frames = {}
for w,h in [(640,480),(1280,720)]:
    y,x = np.indices((h,w))
    source = np.stack([x%256,y%256,(x//16+y//16)%2*180+40],axis=2).astype(np.uint8)
    bank = []
    for i in range(8):
        frame = source.copy()
        frame[:,i*w//8:(i+1)*w//8] = (80,160,240)
        bank.append(frame)
    frames[f'{w}x{h}'] = bank

# A standalone baseline for the requested 1 ms Windows sleep.
sleeps = []
for _ in range(150):
    start = time.perf_counter_ns()
    time.sleep(.001)
    sleeps.append((time.perf_counter_ns()-start)/1e6)
metadata = {'opencv':cv2.__version__, 'gui':'WIN32UI', 'sleep_1ms_actual_ms':stats(sleeps),
            'display_latency_measured':False, 'target_hz':30, 'warmup_frames':30, 'measured_frames':120}
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata),flush=True)

results = []
try:
    for repeat in range(3):
        cases = [(size,mode) for size in frames for mode in ['none','waitkey','pollkey']]
        if repeat == 1:
            cases.reverse()
        elif repeat == 2:
            cases = cases[2:] + cases[:2]
        for size,mode in cases:
            name = f'{repeat+1}_{size}_{mode}'
            if mode != 'none':
                cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(WINDOW,40,40)
            rows = []
            deadline = time.perf_counter_ns()
            measure_cpu_start = measure_wall_start = None
            for index in range(150):
                remaining = (deadline-time.perf_counter_ns())/1e9
                if remaining>0:
                    time.sleep(remaining)
                start = time.perf_counter_ns()
                if index == 30:
                    measure_cpu_start = time.process_time_ns()
                    measure_wall_start = start
                selected = frames[size][index%8]
                before_submit = time.perf_counter_ns()
                if mode != 'none':
                    cv2.imshow(WINDOW,selected)
                submitted = time.perf_counter_ns()
                if mode == 'waitkey':
                    cv2.waitKey(1)
                elif mode == 'pollkey':
                    cv2.pollKey()
                done = time.perf_counter_ns()
                if index>=30:
                    rows.append({'start_ns':start, 'imshow_ms':(submitted-before_submit)/1e6,
                                 'event_pump_ms':(done-submitted)/1e6, 'work_ms':(done-start)/1e6,
                                 'start_lateness_ms':max(0,start-deadline)/1e6})
                deadline += PERIOD_NS
                if deadline < done:
                    deadline = done
            measure_end = time.perf_counter_ns()
            cpu_percent = (time.process_time_ns()-measure_cpu_start)/(measure_end-measure_wall_start)*100
            gaps = np.diff([r['start_ns'] for r in rows])/1e6
            result = {'case':name, 'frames':len(rows),
                      'loop_hz':(len(rows)-1)*1e9/(rows[-1]['start_ns']-rows[0]['start_ns']),
                      'cpu_percent_one_core':cpu_percent,
                      'imshow_ms':stats([r['imshow_ms'] for r in rows]),
                      'event_pump_ms':stats([r['event_pump_ms'] for r in rows]),
                      'work_ms':stats([r['work_ms'] for r in rows]),
                      'start_lateness_ms':stats([r['start_lateness_ms'] for r in rows]),
                      'frame_start_gap_ms':stats(gaps),
                      'work_over_33ms':sum(r['work_ms']>1000/30 for r in rows)}
            results.append(result)
            (OUT/(name+'.json')).write_text(json.dumps({'summary':result,'samples':rows},indent=2))
            (OUT/'summary.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({'case':name,'loop_hz':result['loop_hz'],'imshow_ms':result['imshow_ms'],
                              'event_pump_ms':result['event_pump_ms'],'work_over_33ms':result['work_over_33ms']}),flush=True)
            if mode != 'none':
                cv2.destroyWindow(WINDOW)
finally:
    cv2.destroyAllWindows()
