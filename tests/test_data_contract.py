import unittest

from bankruptcy_prediction.config import DEFAULT_DATA_PATH, TARGET_COLUMN
from bankruptcy_prediction.data import clean_feature_names, load_dataset, make_train_test_split, validate_dataset


class DataContractTest(unittest.TestCase):
    def test_dataset_shape_and_target(self):
        df = clean_feature_names(load_dataset(DEFAULT_DATA_PATH))
        profile = validate_dataset(df)
        self.assertEqual(df.shape[1], 96)
        self.assertEqual(profile["features"], 95)
        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertGreater(profile["bankruptcy_rate"], 0)
        self.assertLess(profile["bankruptcy_rate"], 0.1)

    def test_stratified_split_preserves_classes(self):
        df = clean_feature_names(load_dataset(DEFAULT_DATA_PATH))
        bundle = make_train_test_split(df)
        self.assertEqual(len(bundle.feature_names), 95)
        self.assertEqual(set(bundle.y_train.unique()), {0, 1})
        self.assertEqual(set(bundle.y_test.unique()), {0, 1})


if __name__ == "__main__":
    unittest.main()
