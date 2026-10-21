import unittest

import load_path  # noqa: F401

from src.generate_data import generate_transactions
from src.analyze_data import evaluate_rules


class GenerateTransactionsTests(unittest.TestCase):
    def test_dataset_has_the_expected_columns(self):
        data = generate_transactions(transaction_count=100, seed=7)

        self.assertEqual(len(data), 100)
        self.assertTrue(data["transaction_id"].is_unique)
        expected_cols = {"transaction_id", "timestamp", "sender_id", "receiver_id", "amount", "merchant_category", "location", "device_type", "upi_channel", "anomaly_type", "risk_flag"}
        self.assertTrue(expected_cols.issubset(data.columns))

    def test_dataset_contains_multi_pattern_anomalies(self):
        data = generate_transactions(transaction_count=1_000, seed=7)

        self.assertGreater(data["risk_flag"].sum(), 0)
        # Check that high-value anomalies are >= 15,000
        high_values = data.loc[data["anomaly_type"] == "high_value", "amount"]
        self.assertGreater(len(high_values), 0)
        self.assertTrue((high_values >= 15_000).all())

        anomaly_types = set(data["anomaly_type"].unique())
        self.assertIn("none", anomaly_types)
        self.assertIn("high_value", anomaly_types)
        self.assertIn("velocity_burst", anomaly_types)
        self.assertIn("collect_scam", anomaly_types)
        self.assertIn("odd_hour", anomaly_types)

    def test_velocity_bursts_always_form_detectable_three_transaction_windows(self):
        for seed in (0, 7, 42, 99):
            data = generate_transactions(transaction_count=1_000, seed=seed)
            scored, _ = evaluate_rules(data)
            burst_rows = scored[scored["anomaly_type"] == "velocity_burst"]

            self.assertGreater(len(burst_rows), 0, msg=f"seed={seed}")
            max_window_counts = burst_rows.groupby("sender_id")["rolling_tx_count"].max()
            self.assertTrue((max_window_counts >= 3).all(), msg=f"seed={seed}")


if __name__ == "__main__":
    unittest.main()
