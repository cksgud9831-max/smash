import json
import re
from pathlib import Path

out = Path('/mnt/d/Aiming/results/wsl_generic_environment')
rows = json.loads((out / 'summary.json').read_text())
def times(raw, field):
    result = []
    for block in re.findall(field + r'\s*\{([^}]+)\}', raw):
        sec = re.search(r'\bsec:\s*(\d+)', block)
        nsec = re.search(r'\bnsec:\s*(\d+)', block)
        result.append((int(sec[1]) if sec else 0) + (int(nsec[1]) if nsec else 0) / 1e9)
    return result
for row in rows:
    raw = (out / (row['case'] + '_stats.txt')).read_text()
    sim, real = times(raw, 'sim_time'), times(raw, 'real_time')
    if len(sim) > 1 and len(sim) == len(real):
        row['interval_seconds'] = real[-1] - real[0]
        row['interval_rtf'] = (sim[-1] - sim[0]) / row['interval_seconds']
(out / 'summary.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
