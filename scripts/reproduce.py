"""Bounded offline experiment using the bundled, attributed UCI recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Limit numerical-library threading before importing NumPy/scikit-learn.
# Callers may explicitly override these environment settings.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aqcal.experiment import RunConfig, run_experiment  # noqa: E402
from aqcal.report import format_summary, write_outputs  # noqa: E402

EXPECTED_SHA256 = "13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "quick")
    args = parser.parse_args(argv)
    data = ROOT / "data" / "AirQualityUCI.csv"
    if hashlib.sha256(data.read_bytes()).hexdigest() != EXPECTED_SHA256:
        parser.error("bundled UCI CSV hash differs from the documented upstream recording")
    config = RunConfig(cv_splits=3, search_iterations=2, forest_trees=30,
                       importance_repeats=3, seed=42, n_jobs=1)
    started = time.perf_counter()
    result = run_experiment(str(data), config)
    paths = write_outputs(result, str(args.out))
    elapsed = time.perf_counter() - started
    manifest = {
        "recipe": "python scripts/reproduce.py --out <directory>",
        "dataset_sha256": EXPECTED_SHA256,
        "environment": result.environment,
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "config": result.config,
        "elapsed_seconds": elapsed,
        "output_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for path in paths},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(format_summary(result))
    print(f"\nElapsed: {elapsed:.2f} s; outputs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
