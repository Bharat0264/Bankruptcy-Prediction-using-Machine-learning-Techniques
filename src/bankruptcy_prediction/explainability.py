from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


def model_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> list[dict]:
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        coefficients = getattr(model, "coef_", None)
        if coefficients is None:
            return []
        importances = np.abs(coefficients).ravel()

    rows = [
        {"feature": feature, "importance": float(score)}
        for feature, score in zip(feature_names, importances)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:20]


def permutation_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
) -> list[dict]:
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="average_precision",
        n_jobs=1,
    )
    rows = [
        {"feature": feature, "importance": float(score)}
        for feature, score in zip(feature_names, result.importances_mean)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)[:20]
