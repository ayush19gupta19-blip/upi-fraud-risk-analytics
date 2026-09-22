"""Generate a realistic synthetic UPI transaction dataset with multi-factor risk patterns."""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "upi_transactions.csv"


def _generate_diurnal_timestamps(random: np.random.Generator, count: int, start_date: str = "2026-01-01", days: int = 90) -> pd.Series:
    """Generate timestamps reflecting real-world UPI payment diurnal cycles."""
    # Probability weights for each hour of day (0 to 23)
    # Troughs at 1 AM - 5 AM, peaks around lunch (12-14) and evening (18-22)
    hourly_weights = np.array([
        0.012, 0.006, 0.004, 0.003, 0.005, 0.010,  # 00:00 - 05:00
        0.020, 0.035, 0.055, 0.065, 0.070, 0.075,  # 06:00 - 11:00
        0.080, 0.075, 0.065, 0.060, 0.065, 0.075,  # 12:00 - 17:00
        0.085, 0.090, 0.080, 0.060, 0.040, 0.025   # 18:00 - 23:00
    ])
    hourly_weights /= hourly_weights.sum()

    base_time = pd.Timestamp(start_date)
    random_days = random.integers(0, days, count)
    random_hours = random.choice(24, size=count, p=hourly_weights)
    random_minutes = random.integers(0, 60, count)
    random_seconds = random.integers(0, 60, count)

    time_deltas = (
        pd.to_timedelta(random_days, unit="D")
        + pd.to_timedelta(random_hours, unit="h")
        + pd.to_timedelta(random_minutes, unit="m")
        + pd.to_timedelta(random_seconds, unit="s")
    )
    return base_time + time_deltas


def generate_transactions(transaction_count: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Create normal UPI payments with realistic distributions and injected multi-pattern risk examples."""
    random = np.random.default_rng(seed)
    timestamps = _generate_diurnal_timestamps(random, transaction_count)

    # Base normal amounts: mixture of small everyday transactions and medium ticket sizes
    # Right-skewed distribution resembling retail payment amounts (₹10 - ₹2,500)
    base_amounts = np.round(
        np.clip(random.exponential(scale=350, size=transaction_count) + 20, 10, 2_800), 2
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:06d}" for i in range(1, transaction_count + 1)],
            "timestamp": timestamps,
            "sender_id": [f"USER{i:04d}" for i in random.integers(1, 1001, transaction_count)],
            "receiver_id": [f"REC{i:04d}" for i in random.integers(1, 501, transaction_count)],
            "amount": base_amounts,
            "merchant_category": random.choice(
                ["Food & Dining", "Groceries", "Utilities & Bills", "Shopping", "Travel", "Entertainment"],
                transaction_count,
                p=[0.30, 0.25, 0.15, 0.15, 0.10, 0.05],
            ),
            "location": random.choice(
                ["Delhi", "Mumbai", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata"],
                transaction_count,
            ),
            "device_type": random.choice(["Android", "iOS"], transaction_count, p=[0.76, 0.24]),
            "upi_channel": random.choice(["App", "QR", "Collect Request"], transaction_count, p=[0.55, 0.40, 0.05]),
            "anomaly_type": "none",
            "risk_flag": 0,
        }
    )

    # 1. Injected High-Value Spikes (~1.8% of transactions)
    high_value_count = max(1, round(transaction_count * 0.018))
    high_value_indices = random.choice(transactions.index, size=high_value_count, replace=False)
    transactions.loc[high_value_indices, "amount"] = np.round(random.uniform(15_000, 48_000, high_value_count), 2)
    transactions.loc[high_value_indices, "anomaly_type"] = "high_value"
    transactions.loc[high_value_indices, "risk_flag"] = 1

    # 2. Injected Velocity Bursts (~0.6% burst events: same sender draining account in rapid succession)
    velocity_event_count = max(1, round(transaction_count * 0.002))
    available_indices = transactions[transactions["risk_flag"] == 0].index
    burst_start_indices = random.choice(available_indices, size=velocity_event_count, replace=False)

    for idx in burst_start_indices:
        sender = transactions.at[idx, "sender_id"]
        base_ts = transactions.at[idx, "timestamp"]
        # Create 2 additional rapid payments for this sender within 3-8 minutes
        other_indices = transactions[(transactions["sender_id"] == sender) & (transactions.index != idx)].index
        if len(other_indices) >= 2:
            burst_sub = other_indices[:2]
            for step, sub_idx in enumerate(burst_sub, start=1):
                transactions.at[sub_idx, "timestamp"] = base_ts + pd.Timedelta(minutes=step * 2 + int(random.integers(1, 3)))
                transactions.at[sub_idx, "anomaly_type"] = "velocity_burst"
                transactions.at[sub_idx, "risk_flag"] = 1
            transactions.at[idx, "anomaly_type"] = "velocity_burst"
            transactions.at[idx, "risk_flag"] = 1

    # 3. Injected Collect-Request Phishing Scams (~0.6% high amounts via Collect Request)
    collect_candidates = transactions[(transactions["risk_flag"] == 0) & (transactions["upi_channel"] == "Collect Request")].index
    if len(collect_candidates) > 0:
        sample_size = min(len(collect_candidates), max(1, round(transaction_count * 0.006)))
        scam_indices = random.choice(collect_candidates, size=sample_size, replace=False)
        transactions.loc[scam_indices, "amount"] = np.round(random.uniform(8_000, 35_000, sample_size), 2)
        transactions.loc[scam_indices, "anomaly_type"] = "collect_scam"
        transactions.loc[scam_indices, "risk_flag"] = 1

    # 4. Injected Odd-Hours Drain (~0.5% high transactions between 01:00 AM and 04:30 AM)
    odd_hour_candidates = transactions[transactions["risk_flag"] == 0].index
    if len(odd_hour_candidates) > 0:
        sample_size = min(len(odd_hour_candidates), max(1, round(transaction_count * 0.005)))
        odd_indices = random.choice(odd_hour_candidates, size=sample_size, replace=False)
        for idx in odd_indices:
            current_ts = transactions.at[idx, "timestamp"]
            # Shift time to 02:00 - 04:00 AM
            transactions.at[idx, "timestamp"] = current_ts.replace(hour=int(random.integers(1, 5)), minute=int(random.integers(0, 59)))
            transactions.at[idx, "amount"] = np.round(float(random.uniform(9_000, 32_000)), 2)
            transactions.at[idx, "anomaly_type"] = "odd_hour"
            transactions.at[idx, "risk_flag"] = 1

    return transactions.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    transactions = generate_transactions()
    transactions.to_csv(OUTPUT_FILE, index=False)
    print(f"Created {len(transactions):,} synthetic UPI transactions.")
    print(f"Total risk-injected examples: {transactions['risk_flag'].sum():,}")
    print("Breakdown by anomaly type:")
    print(transactions["anomaly_type"].value_counts().to_string())
    print(f"Saved file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
