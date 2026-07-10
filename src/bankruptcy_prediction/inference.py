from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import METADATA_PATH, MODEL_PATH
from .modeling import probabilities
from .validation import validate_scoring_frame


def load_model(model_path: str | Path = MODEL_PATH, metadata_path: str | Path = METADATA_PATH):
    pipeline = joblib.load(model_path)
    metadata = pd.read_json(metadata_path, typ="series").to_dict()
    return pipeline, metadata


def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [feature for feature in feature_names if feature not in df.columns]
    extra = [feature for feature in df.columns if feature not in feature_names]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if extra:
        df = df.drop(columns=extra)
    return df[feature_names]


def predict_frame(df: pd.DataFrame, pipeline, metadata: dict) -> pd.DataFrame:
    features, _ = validate_scoring_frame(df.copy(), metadata["feature_names"])
    probability = probabilities(pipeline, features)
    threshold = float(metadata["threshold"])
    result = df.copy()
    result["bankruptcy_probability"] = probability
    result["risk_label"] = pd.cut(
        probability,
        bins=[-0.001, 0.2, 0.5, 0.75, 1.001],
        labels=["Low", "Watch", "High", "Critical"],
    ).astype(str)
    result["prediction"] = (probability >= threshold).astype(int)
    return result
