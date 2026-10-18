import unittest
import pandas as pd
from src.analyze_data import (
    RULE_WEIGHTS,
    calculate_iqr_limit,
    compute_rolling_velocity,
    evaluate_rules,
    calculate_metrics,
    explain_transaction,
    train_isolation_forest,
    score_uploaded_transactions,
    validate_transaction_upload,
)


def sample_transactions() -> pd.DataFrame:
    """Small deterministic dataset shared by the explainability and upload tests."""
    return pd.DataFrame(
        {
            "transaction_id": ["TXN001", "TXN002", "TXN003", "TXN004", "TXN005"],
            "timestamp": [
                pd.Timestamp("2026-01-01 10:00:00"),
                pd.Timestamp("2026-01-01 10:02:00"),
                pd.Timestamp("2026-01-01 10:05:00"),  # 3rd txn in 5m for USER001 -> velocity spike
                pd.Timestamp("2026-01-02 03:00:00"),  # Odd hour + high amount + IQR outlier
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

    def test_explain_transaction_reconstructs_composite_score(self):
        scored, upper_limit = evaluate_rules(sample_transactions())
        txn4 = scored.loc[scored["transaction_id"] == "TXN004"].iloc[0]

        explanation = explain_transaction(txn4, iqr_upper_limit=upper_limit)

        self.assertEqual(explanation["transaction_id"], "TXN004")
        self.assertEqual(explanation["risk_score"], int(txn4["risk_score"]))
        self.assertEqual(explanation["risk_level"], "High")

        # Points of the fired rules must reconstruct the composite score exactly
        self.assertEqual(sum(rule["points"] for rule in explanation["rules"]), explanation["risk_score"])

        # Every weighted rule appears exactly once, highest contributor first
        self.assertEqual({rule["column"] for rule in explanation["rules"]}, set(RULE_WEIGHTS))
        self.assertEqual(explanation["rules"][0]["points"], max(RULE_WEIGHTS.values()))

        fired = {rule["column"] for rule in explanation["rules"] if rule["fired"]}
        self.assertEqual(fired, {"iqr_amount_alert", "collect_scam_alert", "odd_hours_alert"})
        self.assertEqual(explanation["fired_rules"], 3)
        self.assertIsNone(explanation["next_tier"])  # 70 pts is already High tier

        # Every fired rule carries a non-empty audit detail
        for rule in explanation["rules"]:
            if rule["fired"]:
                self.assertTrue(rule["detail"])

    def test_explain_transaction_flags_and_next_tier(self):
        scored, _ = evaluate_rules(sample_transactions())

        txn3 = scored.loc[scored["transaction_id"] == "TXN003"].iloc[0]
        explanation3 = explain_transaction(txn3)
        self.assertEqual(explanation3["risk_level"], "Medium")
        self.assertEqual(explanation3["fired_rules"], 1)
        self.assertEqual(explanation3["next_tier"], {"tier": "High", "threshold": 60, "points_needed": 30})

        txn5 = scored.loc[scored["transaction_id"] == "TXN005"].iloc[0]
        explanation5 = explain_transaction(txn5)
        self.assertEqual(explanation5["risk_score"], 0)
        self.assertEqual(explanation5["fired_rules"], 0)
        self.assertTrue(all(rule["points"] == 0 for rule in explanation5["rules"]))
        self.assertEqual(explanation5["next_tier"], {"tier": "Medium", "threshold": 30, "points_needed": 30})

    def test_validate_transaction_upload(self):
        # A fully valid upload reports no problems
        self.assertEqual(validate_transaction_upload(sample_transactions()), [])

        # Missing columns are all reported in one message
        missing = sample_transactions().drop(columns=["amount", "upi_channel"])
        problems = validate_transaction_upload(missing)
        self.assertEqual(len(problems), 1)
        self.assertIn("amount", problems[0])
        self.assertIn("upi_channel", problems[0])

    def test_validate_transaction_upload_bad_values(self):
        bad_amount = sample_transactions()
        bad_amount["amount"] = bad_amount["amount"].astype(str)
        bad_amount.loc[bad_amount.index[0], "amount"] = "not-a-number"
        self.assertTrue(any("'amount'" in problem for problem in validate_transaction_upload(bad_amount)))

        bad_timestamp = sample_transactions()
        bad_timestamp["timestamp"] = bad_timestamp["timestamp"].astype(str)
        bad_timestamp.loc[bad_timestamp.index[0], "timestamp"] = "gibberish"
        self.assertTrue(any("'timestamp'" in problem for problem in validate_transaction_upload(bad_timestamp)))

        empty = sample_transactions().iloc[0:0]
        self.assertTrue(any("no transaction rows" in problem for problem in validate_transaction_upload(empty)))

    def test_validate_transaction_upload_rejects_duplicate_and_empty_ids(self):
        invalid = sample_transactions()
        invalid.loc[1, "transaction_id"] = invalid.loc[0, "transaction_id"]
        invalid.loc[2, "sender_id"] = "  "
        invalid.loc[3, "receiver_id"] = pd.NA

        problems = validate_transaction_upload(invalid)

        self.assertTrue(any("transaction_id" in problem and "duplicate" in problem for problem in problems))
        self.assertTrue(any("sender_id" in problem and "empty" in problem for problem in problems))
        self.assertTrue(any("receiver_id" in problem and "empty" in problem for problem in problems))

    def test_validate_transaction_upload_rejects_invalid_channels_and_mixed_timezones(self):
        invalid = sample_transactions()
        invalid.loc[0, "upi_channel"] = "Bank Transfer"
        invalid["timestamp"] = invalid["timestamp"].astype(object)
        invalid.loc[0, "timestamp"] = "2026-01-01T10:00:00+05:30"

        problems = validate_transaction_upload(invalid)

        self.assertTrue(any("upi_channel" in problem and "invalid" in problem for problem in problems))
        self.assertTrue(any("timestamp" in problem and "timezone" in problem for problem in problems))

    def test_score_uploaded_transactions_accepts_consistent_timezone(self):
        aware = sample_transactions()
        aware["timestamp"] = [
            "2026-01-01T10:00:00+05:30",
            "2026-01-01T04:32:00Z",
            "2026-01-01T10:05:00+05:30",
            "2026-01-01T21:30:00Z",
            "2026-01-02T14:00:00+05:30",
        ]

        scored, _ = score_uploaded_transactions(aware, run_ml=False)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(scored["timestamp"]))
        self.assertIn("risk_score", scored.columns)

    def test_score_uploaded_transactions_runs_full_engine(self):
        scored, upper_limit = score_uploaded_transactions(sample_transactions())

        expected_cols = {
            "transaction_id", "iqr_amount_alert", "velocity_alert", "odd_hours_alert",
            "collect_scam_alert", "risk_score", "risk_level", "review_reason",
            "in_review_queue", "ml_prediction", "ml_anomaly_score",
        }
        self.assertTrue(expected_cols.issubset(scored.columns))
        self.assertEqual(len(scored), 5)
        self.assertGreater(upper_limit, 0)
        self.assertTrue(scored["risk_score"].between(0, 100).all())

        # A high-value odd-hour Collect Request must land in the review queue
        txn4 = scored.loc[scored["transaction_id"] == "TXN004"].iloc[0]
        self.assertEqual(txn4["in_review_queue"], 1)

    def test_score_uploaded_transactions_skips_ml_without_feature_columns(self):
        slim = sample_transactions().drop(columns=["merchant_category", "device_type", "location"])
        scored, _ = score_uploaded_transactions(slim)

        self.assertNotIn("ml_prediction", scored.columns)
        self.assertIn("risk_score", scored.columns)  # rule engine still runs

    def test_score_uploaded_transactions_can_configure_or_skip_ml(self):
        scored, _ = score_uploaded_transactions(
            sample_transactions(),
            run_ml=True,
            contamination=0.2,
            seed=7,
            n_estimators=8,
        )
        self.assertIn("ml_prediction", scored.columns)
        self.assertIn("ml_anomaly_score", scored.columns)

        rules_only, _ = score_uploaded_transactions(sample_transactions(), run_ml=False)
        self.assertNotIn("ml_prediction", rules_only.columns)

    def test_train_isolation_forest_rejects_invalid_estimator_count(self):
        scored, _ = evaluate_rules(sample_transactions())

        with self.assertRaises(ValueError):
            train_isolation_forest(scored, n_estimators=0)

    def test_score_uploaded_transactions_rejects_invalid_upload(self):
        with self.assertRaises(ValueError) as ctx:
            score_uploaded_transactions(sample_transactions().drop(columns=["amount"]))
        self.assertIn("amount", str(ctx.exception))

        with self.assertRaises(ValueError):
            score_uploaded_transactions(sample_transactions().iloc[0:0])

    def test_score_uploaded_transactions_generates_transaction_ids(self):
        scored, _ = score_uploaded_transactions(sample_transactions().drop(columns=["transaction_id"]))
        self.assertIn("transaction_id", scored.columns)
        self.assertTrue(scored["transaction_id"].str.startswith("UPLOAD").all())

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
