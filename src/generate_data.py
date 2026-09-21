"""Create a small, reproducible synthetic UPI transaction dataset.

The labels in this file are deliberately injected for learning and evaluation.
They are not real fraud labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "upi_transactions.csv"
RANDOM_SEED = 42
NUMBER_OF_TRANSACTIONS = 10_000


def build_base_transactions(number_of_transactions: int, seed: int) -> pd.DataFrame:
    """Create ordinary-looking transactions before adding review scenarios."""
    rng = np.random.default_rng(seed)
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 90 * 24 * 60, number_of_transactions), unit="m"
    )

    data = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:06d}" for i in range(1, number_of_transactions + 1)],
            "timestamp": timestamps,
            "sender_id": [f"USER{value:04d}" for value in rng.integers(1, 1001, number_of_transactions)],
            "receiver_id": [f"REC{value:04d}" for value in rng.integers(1, 501, number_of_transactions)],
            "amount": np.round(np.clip(rng.lognormal(mean=6.3, sigma=0.75, size=number_of_transactions), 10, 12_000), 2),
            "merchant_category": rng.choice(
                ["Groceries", "Food", "Travel", "Bills", "Shopping", "Transfer"],
                size=number_of_transactions,
                p=[0.24, 0.23, 0.08, 0.15, 0.17, 0.13],
            ),
            "location": rng.choice(["Delhi", "Mumbai", "Bengaluru", "Pune", "Hyderabad", "Chennai"], size=number_of_transactions),
            "device_type": rng.choice(["Android", "iOS"], size=number_of_transactions, p=[0.76, 0.24]),
            "upi_channel": rng.choice(["App", "QR", "Collect Request"], size=number_of_transactions, p=[0.52, 0.4, 0.08]),
            "risk_flag": 0,
            "risk_scenario": "normal",
        }
    )
    return data.sort_values("timestamp").reset_index(drop=True)


def inject_review_scenarios(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add a few known patterns so students can test their analysis."""
    rng = np.random.default_rng(seed + 1)
    data = data.copy()
    # Three percent keeps the number of review cases small but visible.
    # The minimum of three also lets the function work with tiny test datasets.
    number_of_review_rows = min(len(data), max(3, round(len(data) * 0.03)))
    selected_rows = rng.choice(data.index, size=number_of_review_rows, replace=False)
    high_amount_rows, late_night_rows, repeat_payment_rows = np.array_split(selected_rows, 3)

    data.loc[high_amount_rows, "amount"] = np.round(rng.uniform(15_000, 45_000, len(high_amount_rows)), 2)
    data.loc[high_amount_rows, "risk_scenario"] = "unusually_high_amount"

    data.loc[late_night_rows, "timestamp"] = (
        pd.to_datetime(data.loc[late_night_rows, "timestamp"]).dt.normalize()
        + pd.to_timedelta(rng.integers(0, 5 * 60, len(late_night_rows)), unit="m")
    )
    data.loc[late_night_rows, "risk_scenario"] = "late_night_payment"

    data.loc[repeat_payment_rows, "receiver_id"] = "REC0001"
    data.loc[repeat_payment_rows, "timestamp"] = pd.Timestamp("2026-03-15 14:00:00") + pd.to_timedelta(
        rng.integers(0, 30, len(repeat_payment_rows)), unit="m"
    )
    data.loc[repeat_payment_rows, "risk_scenario"] = "rapid_repeat_payment"

    data.loc[selected_rows, "risk_flag"] = 1
    return data.sort_values("timestamp").reset_index(drop=True)


def create_dataset(number_of_transactions: int = NUMBER_OF_TRANSACTIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build the full reproducible dataset."""
    base_data = build_base_transactions(number_of_transactions, seed)
    return inject_review_scenarios(base_data, seed)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    data = create_dataset()
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Created {len(data):,} transactions at {OUTPUT_FILE}")
    print(data["risk_scenario"].value_counts().to_string())


if __name__ == "__main__":
    main()
