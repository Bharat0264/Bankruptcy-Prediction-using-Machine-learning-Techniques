import unittest

import pandas as pd

from bankruptcy_prediction.config import DEFAULT_DATA_PATH, MAX_ROWS_PER_REQUEST, TARGET_COLUMN
from bankruptcy_prediction.validation import schema_payload, validate_scoring_frame


class ValidationContractTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(DEFAULT_DATA_PATH).drop(columns=[TARGET_COLUMN])
        self.features = self.df.columns.tolist()

    def test_valid_frame_returns_diagnostics(self):
        validated, diagnostics = validate_scoring_frame(self.df.head(3), self.features)
        self.assertEqual(validated.shape, (3, 95))
        self.assertEqual(diagnostics["schema_version"], "bankruptcy-features-v1")

    def test_missing_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_scoring_frame(self.df.drop(columns=[self.features[0]]).head(1), self.features)

    def test_non_numeric_value_is_rejected(self):
        bad = self.df.head(1).copy()
        bad[self.features[0]] = bad[self.features[0]].astype(object)
        bad.loc[bad.index[0], self.features[0]] = "not-a-number"
        with self.assertRaises(ValueError):
            validate_scoring_frame(bad, self.features)

    def test_row_limit_is_enforced(self):
        too_many = pd.concat([self.df.head(1)] * (MAX_ROWS_PER_REQUEST + 1), ignore_index=True)
        with self.assertRaises(ValueError):
            validate_scoring_frame(too_many, self.features)

    def test_schema_payload_exposes_contract(self):
        payload = schema_payload(self.features)
        self.assertEqual(payload["required_column_count"], 95)
        self.assertEqual(payload["max_rows_per_request"], MAX_ROWS_PER_REQUEST)


if __name__ == "__main__":
    unittest.main()
