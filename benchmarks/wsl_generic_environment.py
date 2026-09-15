"""Isolated stock Gazebo environment diagnostic; no project models loaded."""
import os
import subprocess
import signal
import time
import json
import re
import statistics
from pathlib import Path

OUT = Path('/mnt/d/Aiming/results/wsl_generic_environment')
OUT.mkdir(parents=True, exist_ok=True)
WORLD = '/opt/ros/jazzy/opt/gz_sim_vendor/share/gz/gz-sim8/worlds/camera_sensor.sdf'

def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=5)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()

retry = os.environ.get('BENCH_RETRY_CASE')
results = json.loads((OUT / 'summary.json').read_text()) if retry else []
for repeat in range(2):
    cases = [('software', False), ('nvidia', False), ('software', True), ('nvidia', True)]
    if repeat:
        cases.reverse()
    for backend, gui in cases:
        name = f'{repeat+1}_{backend}_{"gui" if gui else "server"}'
        if retry and name != retry:
            continue
        if retry:
            name += '_retry'
        env = os.environ.copy()
        for key in ['LIBGL_ALWAYS_SOFTWARE', 'GALLIUM_DRIVER', 'MESA_D3D12_DEFAULT_ADAPTER_NAME']:
            env.pop(key, None)
        env['GZ_PARTITION'] = 'codex_generic_benchmark_' + name
        if backend == 'software':
            env['LIBGL_ALWAYS_SOFTWARE'] = '1'
        else:
            env['GALLIUM_DRIVER'] = 'd3d12'
            env['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = 'NVIDIA'
        args = ['gz', 'sim', '-r', '-v', '3'] + ([] if gui else ['-s']) + [WORLD]
        print('START ' + name, flush=True)
        with (OUT / (name + '.log')).open('w') as log:
            p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                time.sleep(10)
                raw = subprocess.run(['gz', 'topic', '-e', '-t', '/world/camera_sensor/stats', '-d', '20'], env=env, capture_output=True, text=True, timeout=30)
                (OUT / (name + '_stats.txt')).write_text(raw.stdout + raw.stderr)
                values = [float(v) for v in re.findall(r'real_time_factor:\s*([0-9.eE+-]+)', raw.stdout)]
                row = {'case': name, 'samples': len(values), 'rtf_mean': statistics.mean(values) if values else None, 'rtf_min': min(values) if values else None, 'rtf_max': max(values) if values else None, 'process_running': p.poll() is None}
                results.append(row)
                print(json.dumps(row), flush=True)
                (OUT / 'summary.json').write_text(json.dumps(results, indent=2))
            finally:
                stop(p)
print('FINISHED', flush=True)
