import unittest
import pandas as pd
from src.analyze_data import calculate_iqr_limit, compute_rolling_velocity, evaluate_rules, calculate_metrics


class AnalyzeDataTests(unittest.TestCase):
    def test_calculate_iqr_limit(self):
        # Q1 = 25, Q3 = 75, IQR = 50 -> limit = 75 + 1.5 * 50 = 150
        series = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90])
        limit = calculate_iqr_limit(series, multiplier=1.5)
        self.assertGreater(limit, series.quantile(0.75))

        # Empty series test
        empty_limit = calculate_iqr_limit(pd.Series([], dtype=float))
        self.assertEqual(empty_limit, 0.0)

    def test_compute_rolling_velocity(self):
        df = pd.DataFrame(
            {
                "sender_id": ["USER_A", "USER_B", "USER_A", "USER_A"],
                "timestamp": [
                    pd.Timestamp("2026-01-01 10:00:00"),
                    pd.Timestamp("2026-01-01 10:01:00"),
                    pd.Timestamp("2026-01-01 10:04:00"),
                    pd.Timestamp("2026-01-01 10:25:00"),  # > 15m later
                ],
            }
        )
        counts = compute_rolling_velocity(df, window_mins=15)
        self.assertEqual(list(counts), [1, 1, 2, 1])

    def test_multi_factor_rules(self):
        # Create a small deterministic dataset
        data = pd.DataFrame(
            {
                "transaction_id": ["TXN001", "TXN002", "TXN003", "TXN004", "TXN005"],
                "timestamp": [
                    pd.Timestamp("2026-01-01 10:00:00"),
                    pd.Timestamp("2026-01-01 10:02:00"),
                    pd.Timestamp("2026-01-01 10:05:00"),  # 3rd txn in 5m for USER001 -> velocity spike
                    pd.Timestamp("2026-01-02 03:00:00"),  # Odd hour + high amount
                    pd.Timestamp("2026-01-02 14:00:00"),  # Normal
                ],
                "sender_id": ["USER001", "USER001", "USER001", "USER002", "USER003"],
                "receiver_id": ["REC001", "REC002", "REC003", "REC004", "REC005"],
                "amount": [500.0, 600.0, 700.0, 5000.0, 200.0],
                "merchant_category": ["Food & Dining", "Shopping", "Food & Dining", "Travel", "Groceries"],
                "location": ["Delhi", "Delhi", "Delhi", "Mumbai", "Pune"],
                "device_type": ["Android", "Android", "Android", "iOS", "Android"],
                "upi_channel": ["App", "App", "App", "Collect Request", "QR"],
                "risk_flag": [0, 0, 1, 1, 0],
            }
        )

        enriched, upper_limit = evaluate_rules(data, velocity_window_mins=15, velocity_threshold=3)

        # Check column additions
        expected_cols = {
            "iqr_amount_alert", "rolling_tx_count", "velocity_alert",
            "odd_hours_alert", "collect_scam_alert", "risk_score",
            "risk_level", "review_reason", "in_review_queue"
        }
        self.assertTrue(expected_cols.issubset(enriched.columns))

        # Check velocity trigger for TXN003
        txn3 = enriched.loc[enriched["transaction_id"] == "TXN003"].iloc[0]
        self.assertEqual(txn3["velocity_alert"], 1)
        self.assertGreaterEqual(txn3["rolling_tx_count"], 3)

        # Check odd hours & collect scam trigger for TXN004
        txn4 = enriched.loc[enriched["transaction_id"] == "TXN004"].iloc[0]
        self.assertEqual(txn4["odd_hours_alert"], 1)
        self.assertEqual(txn4["collect_scam_alert"], 1)

        # Check risk score bounds
        self.assertTrue((enriched["risk_score"] >= 0).all() and (enriched["risk_score"] <= 100).all())

    def test_calculate_metrics(self):
        y_true = pd.Series([1, 1, 0, 0, 1])
        y_pred = pd.Series([1, 0, 0, 1, 1])
        # TP: 2, FP: 1, FN: 1, TN: 1 -> Precision: 2/3, Recall: 2/3
        metrics = calculate_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["precision"], 2 / 3, places=3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3, places=3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3, places=3)


if __name__ == "__main__":
    unittest.main()
