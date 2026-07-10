from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DRIFT_BASELINE_PATH
from .validation import validate_scoring_frame


def build_baseline(df: pd.DataFrame, feature_names: list[str], path: Path = DRIFT_BASELINE_PATH) -> None:
    features = df[feature_names]
    profile = {
        "feature_count": len(feature_names),
        "row_count": int(len(features)),
        "features": {},
    }
    for feature in feature_names:
        series = pd.to_numeric(features[feature], errors="coerce")
        profile["features"][feature] = {
            "mean": float(series.mean()),
            "std": float(series.std() or 0.0),
            "min": float(series.min()),
            "max": float(series.max()),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_baseline(path: Path = DRIFT_BASELINE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError("Drift baseline was not found. Retrain the model to generate it.")
    return json.loads(path.read_text(encoding="utf-8"))


def drift_report(df: pd.DataFrame, feature_names: list[str], baseline: dict | None = None) -> dict:
    validated, diagnostics = validate_scoring_frame(df, feature_names)
    reference = baseline or load_baseline()
    rows = []
    for feature in feature_names:
        current = validated[feature]
        stats = reference["features"][feature]
        current_mean = float(current.mean())
        std = float(stats["std"]) or 1.0
        z_shift = abs(current_mean - float(stats["mean"])) / std
        out_of_range_rate = float(((current < stats["min"]) | (current > stats["max"])).mean())
        severity = "ok"
        if z_shift >= 3 or out_of_range_rate >= 0.25:
            severity = "high"
        elif z_shift >= 1.5 or out_of_range_rate >= 0.1:
            severity = "watch"
        rows.append(
            {
                "feature": feature,
                "current_mean": current_mean,
                "baseline_mean": float(stats["mean"]),
                "z_shift": float(z_shift),
                "out_of_range_rate": out_of_range_rate,
                "severity": severity,
            }
        )
    high = sum(1 for row in rows if row["severity"] == "high")
    watch = sum(1 for row in rows if row["severity"] == "watch")
    return {
        "schema_version": diagnostics["schema_version"],
        "rows_checked": diagnostics["rows"],
        "baseline_rows": reference["row_count"],
        "high_drift_features": high,
        "watch_drift_features": watch,
        "overall_status": "high" if high else "watch" if watch else "ok",
        "top_features": sorted(rows, key=lambda row: (row["severity"] != "high", -row["z_shift"]))[:20],
    }
