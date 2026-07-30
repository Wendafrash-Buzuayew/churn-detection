import unittest

import pandas as pd


class HybridFeatureBuilderTests(unittest.TestCase):
    def test_feature_builder_creates_terminal_multi_service_collapse(self):
        from unified_churn_engine import HybridFeatureBuilder
        df = pd.DataFrame({
            "AON": [500],
            "DATA_MB_W10": [100], "DATA_MB_W11": [100], "DATA_MB_W12": [100], "DATA_MB_W13": [0],
            "OG_VOICE_MIN_W10": [10], "OG_VOICE_MIN_W11": [10], "OG_VOICE_MIN_W12": [10], "OG_VOICE_MIN_W13": [0],
            "BUNDLE_CNT_W10": [1], "BUNDLE_CNT_W11": [1], "BUNDLE_CNT_W12": [1], "BUNDLE_CNT_W13": [0],
        })
        result = HybridFeatureBuilder().transform(df)
        self.assertEqual(result.loc[0, "RULE_TERMINAL_MULTI_SERVICE"], 1)

    def test_terminal_zero_run_counts_only_trailing_inactivity(self):
        from unified_churn_engine import HybridFeatureBuilder
        df = pd.DataFrame({"DATA_MB_W10": [100], "DATA_MB_W11": [0], "DATA_MB_W12": [0], "DATA_MB_W13": [0]})
        result = HybridFeatureBuilder().transform(df)
        self.assertEqual(result.loc[0, "DATA_TERMINAL_ZERO_RUN"], 3)


if __name__ == "__main__":
    unittest.main()

class UnifiedEngineIntegrationTests(unittest.TestCase):
    @staticmethod
    def _labeled_population():
        rows = []
        for index in range(80):
            churn = index % 4 == 0
            row = {"DATASET_TYPE": "TRAIN" if index < 60 else "TEST", "LABEL_CHURN_30D": int(churn), "LABEL_CHURN_90D": int(churn), "AON": 400}
            for prefix, value in (("DATA_MB", 100.0), ("OG_VOICE_MIN", 10.0), ("BUNDLE_CNT", 2.0), ("TOTAL_REVENUE", 20.0)):
                for week in (10, 11, 12): row[f"{prefix}_W{week}"] = value
                row[f"{prefix}_W13"] = 0.0 if churn else value
            rows.append(row)
        return pd.DataFrame(rows)

    def test_training_excludes_all_labels_and_scores_unlabeled_oot(self):
        from unified_churn_engine import EngineConfig, UnifiedChurnEngine
        engine = UnifiedChurnEngine(EngineConfig(cv_folds=2, max_iter=20, precision_floor=.5))
        summary = engine.fit(self._labeled_population())
        self.assertEqual(summary["target"], "LABEL_CHURN_30D")
        self.assertTrue(all(not name.startswith("LABEL_CHURN_") for name in engine.feature_columns_))
        oot = self._labeled_population().drop(columns=["LABEL_CHURN_30D", "LABEL_CHURN_90D"])
        scored = engine.score(oot)
        self.assertIn("churn_probability", scored)
        self.assertIn("false_positive_suppressed", scored)
