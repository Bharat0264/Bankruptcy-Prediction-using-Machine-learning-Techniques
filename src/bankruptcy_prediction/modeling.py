from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float | int]
    threshold: float


def _xgboost_model() -> Any | None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    return XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def candidate_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    xgb_model = _xgboost_model()
    if xgb_model is not None:
        models["XGBoost"] = xgb_model
    return models


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def probabilities(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    scores = pipeline.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def choose_threshold(y_true: pd.Series, y_prob: np.ndarray, min_recall: float = 0.72) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        precision = precision_score(y_true, predictions, zero_division=0)
        score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
        if recall >= min_recall and score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    y_prob = probabilities(pipeline, X_test)
    threshold = choose_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return ModelResult(name=name, pipeline=pipeline, metrics=metrics, threshold=threshold)


def train_and_select(bundle) -> tuple[ModelResult, list[ModelResult]]:
    results: list[ModelResult] = []
    for name, model in candidate_models().items():
        pipeline = make_pipeline(model)
        pipeline.fit(bundle.X_train, bundle.y_train)
        results.append(evaluate_model(name, pipeline, bundle.X_test, bundle.y_test))

    best = max(
        results,
        key=lambda result: (
            float(result.metrics["average_precision"]),
            float(result.metrics["recall"]),
            float(result.metrics["balanced_accuracy"]),
        ),
    )
    return best, results
