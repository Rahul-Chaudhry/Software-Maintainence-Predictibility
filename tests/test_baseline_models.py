import unittest

from Code.run_baseline_models import FEATURES, evaluate, load_dataset


class BaselineModelTests(unittest.TestCase):
    def test_included_dataset_has_expected_columns(self) -> None:
        dataframe = load_dataset("QUES")

        self.assertTrue(set(FEATURES).issubset(dataframe.columns))
        self.assertIn("change", dataframe.columns)

    def test_evaluation_returns_metrics_for_every_model(self) -> None:
        results = evaluate("UIMS")

        self.assertEqual(len(results), 5)
        self.assertEqual(set(results["Dataset"]), {"UIMS"})
        self.assertFalse(results[["R2", "MAE", "RMSE"]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
