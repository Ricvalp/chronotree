Sono a:

'''
Wait for both to finish. Check their completion records:
'''

da runnare:

'''
.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("runs/router_utility_v1")
for name, path in [
    ("Sampler audit", root / "sampler_audit/summary.json"),
    ("Smoke check", root / "measurements/S0/step_00090000/smoke/train/worker_0.json"),
]:
    result = json.loads(path.read_text())
    print(name, result["status"])
    assert result["status"] == "complete"
PY
'''