import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_metric_clis_have_help():
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for name in ("compute_layer_similarity.py", "fit_pairwise_mapping.py", "evaluate_patch_alignment.py"):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"], env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
