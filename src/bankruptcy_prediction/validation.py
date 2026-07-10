from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_ROWS_PER_REQUEST


def validate_scoring_frame(df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        raise ValueError("No rows were provided for scoring.")
    if len(df) > MAX_ROWS_PER_REQUEST:
        raise ValueError(f"Scoring is limited to {MAX_ROWS_PER_REQUEST} rows per request.")

    missing = [feature for feature in feature_names if feature not in df.columns]
    extra = [feature for feature in df.columns if feature not in feature_names]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"Missing {len(missing)} required columns: {preview}{suffix}")

    features = df[feature_names].copy()
    non_numeric: list[str] = []
    for column in feature_names:
        converted = pd.to_numeric(features[column], errors="coerce")
        if converted.isna().any() and not features[column].isna().any():
            non_numeric.append(column)
        features[column] = converted

    if non_numeric:
        preview = ", ".join(non_numeric[:8])
        suffix = "..." if len(non_numeric) > 8 else ""
        raise ValueError(f"Non-numeric values found in columns: {preview}{suffix}")

    missing_cells = int(features.isna().sum().sum())
    if missing_cells:
        raise ValueError(f"Input contains {missing_cells} missing numeric values.")

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Input contains infinite values.")

    diagnostics = {
        "rows": int(len(features)),
        "columns": int(len(features.columns)),
        "extra_columns_ignored": extra,
        "schema_version": "bankruptcy-features-v1",
    }
    return features, diagnostics


def schema_payload(feature_names: list[str]) -> dict:
    return {
        "schema_version": "bankruptcy-features-v1",
        "target": "Bankrupt?",
        "required_columns": feature_names,
        "required_column_count": len(feature_names),
        "max_rows_per_request": MAX_ROWS_PER_REQUEST,
        "types": {feature: "number" for feature in feature_names},
    }
