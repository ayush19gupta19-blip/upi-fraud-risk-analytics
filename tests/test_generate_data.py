import unittest

from src.generate_data import generate_transactions


class GenerateTransactionsTests(unittest.TestCase):
    def test_dataset_has_the_expected_columns(self):
        data = generate_transactions(transaction_count=100, seed=7)

        self.assertEqual(len(data), 100)
        self.assertTrue(data["transaction_id"].is_unique)
        self.assertTrue({"transaction_id", "amount", "risk_flag"}.issubset(data.columns))

    def test_dataset_contains_high_value_examples(self):
        data = generate_transactions(transaction_count=1_000, seed=7)

        self.assertGreater(data["risk_flag"].sum(), 0)
        self.assertTrue((data.loc[data["risk_flag"] == 1, "amount"] >= 15_000).all())


if __name__ == "__main__":
    unittest.main()
