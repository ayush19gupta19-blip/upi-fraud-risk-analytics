import unittest
from src.generate_data import generate_transactions


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

        # Check anomaly types present
        anomaly_types = set(data["anomaly_type"].unique())
        self.assertTrue({"none", "high_value"}.issubset(anomaly_types))


if __name__ == "__main__":
    unittest.main()
