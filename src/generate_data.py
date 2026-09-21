"""Generate a small synthetic UPI transaction dataset for this project."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "upi_transactions.csv"


def generate_transactions(transaction_count: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Create normal payments, then add a few known high-value examples."""
    random = np.random.default_rng(seed)
    timestamps = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        random.integers(0, 90 * 24 * 60, transaction_count), unit="m"
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:06d}" for i in range(1, transaction_count + 1)],
            "timestamp": timestamps,
            "sender_id": [f"USER{i:04d}" for i in random.integers(1, 1001, transaction_count)],
            "receiver_id": [f"REC{i:04d}" for i in random.integers(1, 501, transaction_count)],
            "amount": np.round(np.clip(random.normal(650, 250, transaction_count), 50, 2_000), 2),
            "merchant_category": random.choice(["Food", "Groceries", "Bills", "Shopping", "Travel"], transaction_count),
            "location": random.choice(["Delhi", "Mumbai", "Bengaluru", "Pune", "Hyderabad"], transaction_count),
            "device_type": random.choice(["Android", "iOS"], transaction_count, p=[0.75, 0.25]),
            "upi_channel": random.choice(["App", "QR", "Collect Request"], transaction_count, p=[0.55, 0.4, 0.05]),
            "risk_flag": 0,
        }
    )

    # These are deliberately high payments used only to check the demo rule.
    high_risk_count = max(1, round(transaction_count * 0.03))
    high_risk_rows = random.choice(transactions.index, size=high_risk_count, replace=False)
    transactions.loc[high_risk_rows, "amount"] = np.round(random.uniform(15_000, 45_000, high_risk_count), 2)
    transactions.loc[high_risk_rows, "risk_flag"] = 1

    return transactions.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    transactions = generate_transactions()
    transactions.to_csv(OUTPUT_FILE, index=False)
    print(f"Created {len(transactions):,} synthetic transactions.")
    print(f"Known high-value examples added: {transactions['risk_flag'].sum():,}")
    print(f"Saved file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
