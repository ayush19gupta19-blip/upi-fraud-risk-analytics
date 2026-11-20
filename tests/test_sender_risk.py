import unittest

import load_path  # noqa: F401

import pandas as pd

from src.sender_risk import (
    annotate_sender_amount_anomalies,
    build_sender_profiles,
    build_sender_risk_export,
    detect_high_fan_in_receivers,
    rank_sender_risk,
)


def _sample_frame() -> pd.DataFrame:
    """Deterministic mix of steady payers, a burst outlier, and a fan-in hub."""
    rows = []
    # USER_A: five modest App payments + one huge personal outlier
    base_times = pd.date_range("2026-03-01 10:00", periods=5, freq="1h")
    for i, ts in enumerate(base_times):
        rows.append(
            {
                "transaction_id": f"A{i}",
                "timestamp": ts,
                "sender_id": "USER_A",
                "receiver_id": f"REC_{i % 2}",
                "amount": 500.0 + i * 10,
                "upi_channel": "App",
                "risk_score": 0,
                "in_review_queue": 0,
            }
        )
    rows.append(
        {
            "transaction_id": "A_OUTLIER",
            "timestamp": pd.Timestamp("2026-03-01 16:00"),
            "sender_id": "USER_A",
            "receiver_id": "REC_X",
            "amount": 50_000.0,
            "upi_channel": "App",
            "risk_score": 40,
            "in_review_queue": 1,
        }
    )
    # USER_B: single small payment (too little history for personal fence)
    rows.append(
        {
            "transaction_id": "B0",
            "timestamp": pd.Timestamp("2026-03-02 12:00"),
            "sender_id": "USER_B",
            "receiver_id": "REC_Y",
            "amount": 200.0,
            "upi_channel": "QR",
            "risk_score": 0,
            "in_review_queue": 0,
        }
    )
    # Many senders pay the same receiver (fan-in)
    for i in range(10):
        rows.append(
            {
                "transaction_id": f"F{i}",
                "timestamp": pd.Timestamp("2026-03-03 09:00") + pd.Timedelta(minutes=i),
                "sender_id": f"FAN_{i}",
                "receiver_id": "MULE_1",
                "amount": 1_000.0,
                "upi_channel": "Collect Request",
                "risk_score": 20,
                "in_review_queue": 0,
            }
        )
    return pd.DataFrame(rows)


class SenderRiskTests(unittest.TestCase):
    def test_build_sender_profiles_aggregates_history(self):
        profiles = build_sender_profiles(_sample_frame())
        self.assertIn("USER_A", set(profiles["sender_id"]))
        user_a = profiles.loc[profiles["sender_id"] == "USER_A"].iloc[0]
        self.assertEqual(int(user_a["txn_count"]), 6)
        self.assertGreater(float(user_a["personal_upper_fence"]), float(user_a["median_amount"]))
        self.assertTrue(0 <= float(user_a["receiver_concentration"]) <= 1)

    def test_annotate_sender_amount_anomalies_flags_personal_outliers(self):
        frame = _sample_frame()
        annotated = annotate_sender_amount_anomalies(frame)
        outlier = annotated.loc[annotated["transaction_id"] == "A_OUTLIER"].iloc[0]
        self.assertEqual(int(outlier["sender_amount_anomaly"]), 1)
        self.assertGreater(float(outlier["amount_vs_sender_median"]), 10)
        # Thin history should not fire
        thin = annotated.loc[annotated["transaction_id"] == "B0"].iloc[0]
        self.assertEqual(int(thin["sender_amount_anomaly"]), 0)

    def test_detect_high_fan_in_receivers(self):
        receivers = detect_high_fan_in_receivers(_sample_frame(), min_unique_senders=8)
        mule = receivers.loc[receivers["receiver_id"] == "MULE_1"].iloc[0]
        self.assertEqual(int(mule["unique_senders"]), 10)
        self.assertEqual(int(mule["high_fan_in"]), 1)

    def test_rank_sender_risk_orders_by_index(self):
        annotated, leaderboard, receivers = build_sender_risk_export(_sample_frame())
        self.assertTrue(leaderboard["sender_risk_index"].is_monotonic_decreasing)
        self.assertIn("sender_risk_tier", leaderboard.columns)
        self.assertGreater(len(leaderboard), 0)
        self.assertIn("sender_amount_anomaly", annotated.columns)
        self.assertGreaterEqual(int(receivers["high_fan_in"].sum()), 1)

    def test_rank_sender_risk_handles_missing_rule_columns(self):
        bare = _sample_frame().drop(columns=["risk_score", "in_review_queue"])
        ranked = rank_sender_risk(bare)
        self.assertEqual(len(ranked), bare["sender_id"].nunique())
        self.assertTrue((ranked["sender_risk_index"] >= 0).all())

    def test_rejects_negative_thresholds(self):
        frame = _sample_frame()
        with self.assertRaisesRegex(ValueError, "min_history"):
            annotate_sender_amount_anomalies(frame, min_history=0)
        with self.assertRaisesRegex(ValueError, "min_unique_senders"):
            detect_high_fan_in_receivers(frame, min_unique_senders=0)

    def test_empty_frame_returns_empty_schemas(self):
        empty = pd.DataFrame(
            columns=["sender_id", "amount", "timestamp", "receiver_id", "upi_channel"]
        )
        profiles = build_sender_profiles(empty)
        self.assertEqual(len(profiles), 0)
        self.assertIn("personal_upper_fence", profiles.columns)
        receivers = detect_high_fan_in_receivers(empty)
        self.assertEqual(len(receivers), 0)


if __name__ == "__main__":
    unittest.main()
