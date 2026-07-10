from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TARGET_COLUMN


@dataclass(frozen=True)
class DataBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def load_dataset(path: str | Path, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in {path}.")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    return cleaned


def validate_dataset(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> dict:
    feature_df = df.drop(columns=[target_column])
    non_numeric = feature_df.select_dtypes(exclude="number").columns.tolist()
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    class_counts = df[target_column].value_counts().sort_index().to_dict()

    if non_numeric:
        raise ValueError(f"Non-numeric feature columns are not supported: {non_numeric}")
    if set(class_counts) - {0, 1}:
        raise ValueError("Target must contain only 0 and 1 labels.")

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "features": int(feature_df.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "bankruptcy_rate": float(df[target_column].mean()),
    }


def make_train_test_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> DataBundle:
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return DataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=X.columns.tolist(),
    )


def write_prediction_template(df: pd.DataFrame, path: str | Path, target_column: str = TARGET_COLUMN) -> None:
    template = df.drop(columns=[target_column]).head(5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
