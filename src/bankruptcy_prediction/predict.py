from __future__ import annotations

import argparse

import pandas as pd

from .inference import load_model, predict_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score companies with the trained bankruptcy model.")
    parser.add_argument("--input", required=True, help="CSV with the 95 model features.")
    parser.add_argument("--output", default="reports/predictions.csv", help="Where to write scored rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline, metadata = load_model()
    df = pd.read_csv(args.input)
    predictions = predict_frame(df, pipeline, metadata)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} scored rows to {args.output}")


if __name__ == "__main__":
    main()
