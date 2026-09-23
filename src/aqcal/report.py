"""Writing experiment results to disk and formatting the console summary."""

from __future__ import annotations

import json
import math
import os

from aqcal.experiment import ExperimentResult

METRICS_JSON = "metrics.json"
METRICS_CSV = "metrics.csv"
IMPORTANCE_CSV = "feature_importance.csv"


def _finite_or_none(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean(obj):
    if isinstance(obj, dict):
        return {key: _clean(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(value) for value in obj]
    return _finite_or_none(obj)


def to_json_dict(result: ExperimentResult) -> dict:
    return _clean(
        {
            "dataset": result.dataset,
            "periods": result.periods,
            "tuning": result.tuning,
            "selection_rule": "lowest validation RMSE; test block used only for final scores",
            "selected_model": result.selected_model,
            "scores": result.scores,
            "permutation_importance": {
                "model": result.selected_model,
                "block": "test",
                "measure": "increase in RMSE when the feature is shuffled",
                "values": result.importance.to_dict(orient="records"),
            },
            "config": result.config,
        }
    )


def write_outputs(result: ExperimentResult, out_dir: str) -> list[str]:
    """Write metrics.json, metrics.csv and feature_importance.csv; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, name) for name in (METRICS_JSON, METRICS_CSV, IMPORTANCE_CSV)]

    with open(paths[0], "w", encoding="utf-8") as handle:
        json.dump(to_json_dict(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    result.scores_table().to_csv(paths[1], index=False, float_format="%.6f")
    result.importance.to_csv(paths[2], index=False, float_format="%.6f")
    return paths


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def format_summary(result: ExperimentResult) -> str:
    ds = result.dataset
    lines = [
        f"Target: {ds['target']} (mg/m^3)",
        (
            f"Rows parsed: {ds['rows_parsed']}; dropped without target: {ds['rows_missing_target']}; "
            f"dropped without sensor data: {ds['rows_without_sensor_data']}; used: {ds['rows_used']}"
        ),
    ]
    for block, info in result.periods.items():
        lines.append(f"  {block:<10} {info['rows']:>5} rows  {info['start']} .. {info['end']}")

    ridge_info = result.tuning["ridge"]
    hgb_info = result.tuning["hist_gradient_boosting"]
    lines.append(f"Tuning: {result.tuning['cv']}")
    lines.append(f"  ridge alpha = {ridge_info['alpha']:g} (CV RMSE {_fmt(ridge_info['cv_rmse'])})")
    params = ", ".join(
        f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in hgb_info["params"].items()
    )
    lines.append(f"  boosting: {params} (CV RMSE {_fmt(hgb_info['cv_rmse'])})")

    header = f"{'model':<24}{'val RMSE':>10}{'test RMSE':>11}{'test MAE':>10}{'test R2':>9}"
    lines += ["", header, "-" * len(header)]
    for row in result.scores_table().itertuples(index=False):
        marker = "  <- selected" if row.model == result.selected_model else ""
        lines.append(
            f"{row.model:<24}{_fmt(row.validation_rmse):>10}{_fmt(row.test_rmse):>11}"
            f"{_fmt(row.test_mae):>10}{_fmt(row.test_r2):>9}{marker}"
        )

    lines += ["", f"Permutation importance of {result.selected_model} on the test block (RMSE increase):"]
    for row in result.importance.itertuples(index=False):
        lines.append(f"  {row.feature:<16}{_fmt(row.rmse_increase_mean):>8} +/- {_fmt(row.rmse_increase_std)}")
    return "\n".join(lines)
