import unittest

import pandas as pd

from bankruptcy_prediction.config import DEFAULT_DATA_PATH, TARGET_COLUMN
from bankruptcy_prediction.inference import align_features


class InferenceContractTest(unittest.TestCase):
    def test_align_features_reorders_and_drops_extra_columns(self):
        df = pd.read_csv(DEFAULT_DATA_PATH).drop(columns=[TARGET_COLUMN]).head(2)
        feature_names = df.columns.tolist()
        shuffled = df[list(reversed(feature_names))].copy()
        shuffled["extra_column"] = 1
        aligned = align_features(shuffled, feature_names)
        self.assertEqual(aligned.columns.tolist(), feature_names)
        self.assertNotIn("extra_column", aligned.columns)

    def test_align_features_rejects_missing_columns(self):
        df = pd.read_csv(DEFAULT_DATA_PATH).drop(columns=[TARGET_COLUMN]).head(2)
        feature_names = df.columns.tolist()
        with self.assertRaises(ValueError):
            align_features(df.drop(columns=[feature_names[0]]), feature_names)


if __name__ == "__main__":
    unittest.main()
