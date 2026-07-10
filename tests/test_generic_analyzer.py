import unittest

import pandas as pd

from bankruptcy_prediction.generic_analyzer import analyze_company_csv


class GenericAnalyzerTest(unittest.TestCase):
    def test_sales_data_gets_risk_analysis(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
                "product": ["A", "A", "B", "B"],
                "sales": [10000, 9000, 7600, 7000],
                "profit": [1400, 900, 250, -100],
                "expense": [8600, 8100, 7350, 7100],
            }
        )
        result = analyze_company_csv(df)
        self.assertEqual(result["mode"], "generic_company_analysis")
        self.assertIn(result["risk_label"], {"Low", "Watch", "High", "Critical"})
        self.assertGreaterEqual(result["risk_score"], 0)
        self.assertGreater(len(result["recommendations"]), 0)

    def test_non_numeric_company_data_is_rejected(self):
        df = pd.DataFrame({"company": ["A", "B"], "region": ["East", "West"]})
        with self.assertRaises(ValueError):
            analyze_company_csv(df)


if __name__ == "__main__":
    unittest.main()
