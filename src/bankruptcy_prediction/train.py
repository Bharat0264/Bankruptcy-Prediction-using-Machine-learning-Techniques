from __future__ import annotations

import argparse

from .config import DEFAULT_DATA_PATH, PREDICTION_TEMPLATE_PATH
from .data import clean_feature_names, load_dataset, make_train_test_split, validate_dataset, write_prediction_template
from .explainability import model_feature_importance, permutation_feature_importance
from .modeling import train_and_select
from .reporting import write_artifacts, write_drift_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bankruptcy risk model suite.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to the training CSV.")
    parser.add_argument("--quick", action="store_true", help="Skip permutation importance for faster local runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = clean_feature_names(load_dataset(args.data))
    profile = validate_dataset(df)
    bundle = make_train_test_split(df)
    best, results = train_and_select(bundle)
    feature_importance = model_feature_importance(best.pipeline, bundle.feature_names)
    permutation = [] if args.quick else permutation_feature_importance(
        best.pipeline,
        bundle.X_test,
        bundle.y_test,
        bundle.feature_names,
    )
    write_artifacts(best, results, profile, bundle.feature_names, feature_importance, permutation)
    write_drift_baseline(df, bundle.feature_names)
    write_prediction_template(df, PREDICTION_TEMPLATE_PATH)

    print(f"Selected model: {best.name}")
    print(f"Average precision: {best.metrics['average_precision']:.3f}")
    print(f"Recall: {best.metrics['recall']:.3f}")
    print(f"ROC-AUC: {best.metrics['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
