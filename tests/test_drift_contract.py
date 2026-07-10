import unittest

import pandas as pd

from bankruptcy_prediction.config import DEFAULT_DATA_PATH, TARGET_COLUMN
from bankruptcy_prediction.drift import drift_report


class DriftContractTest(unittest.TestCase):
    def test_drift_report_returns_status(self):
        df = pd.read_csv(DEFAULT_DATA_PATH)
        feature_names = df.drop(columns=[TARGET_COLUMN]).columns.tolist()
        sample = df.drop(columns=[TARGET_COLUMN]).head(10)
        baseline = {
            "feature_count": len(feature_names),
            "row_count": int(len(df)),
            "features": {},
        }
        for feature in feature_names:
            series = df[feature]
            baseline["features"][feature] = {
                "mean": float(series.mean()),
                "std": float(series.std() or 0.0),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        report = drift_report(sample, feature_names, baseline=baseline)
        self.assertIn(report["overall_status"], {"ok", "watch", "high"})
        self.assertEqual(report["rows_checked"], 10)
        self.assertGreater(len(report["top_features"]), 0)


if __name__ == "__main__":
    unittest.main()
