from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .config import METADATA_PATH, MODEL_PATH, REPORT_PATH
from .drift import build_baseline


def result_table(results) -> list[dict]:
    rows = []
    for result in results:
        row = {"model": result.name}
        row.update(result.metrics)
        rows.append(row)
    return sorted(rows, key=lambda row: row["average_precision"], reverse=True)


def write_artifacts(
    best,
    results,
    dataset_profile: dict,
    feature_names: list[str],
    feature_importance: list[dict],
    permutation_importance: list[dict],
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    report_path: Path = REPORT_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.pipeline, model_path)

    metadata = {
        "model_name": best.name,
        "threshold": best.threshold,
        "feature_names": feature_names,
        "target_column": "Bankrupt?",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = {
        "generated_at": metadata["created_at"],
        "selected_model": best.name,
        "selected_metrics": best.metrics,
        "dataset_profile": dataset_profile,
        "leaderboard": result_table(results),
        "feature_importance": feature_importance,
        "permutation_importance": permutation_importance,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["leaderboard"]).to_csv(report_path.with_suffix(".csv"), index=False)


def write_drift_baseline(df: pd.DataFrame, feature_names: list[str]) -> None:
    build_baseline(df, feature_names)
