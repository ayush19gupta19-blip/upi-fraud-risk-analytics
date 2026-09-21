"""Use one simple IQR rule to create a UPI transaction review queue."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "upi_transactions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "transactions_with_iqr_flags.csv"


def add_iqr_alerts(transactions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Alert when an amount is above Q3 + 1.5 times IQR."""
    first_quartile = transactions["amount"].quantile(0.25)
    third_quartile = transactions["amount"].quantile(0.75)
    iqr = third_quartile - first_quartile
    upper_limit = third_quartile + 1.5 * iqr

    transactions = transactions.copy()
    transactions["iqr_amount_alert"] = (transactions["amount"] > upper_limit).astype(int)
    transactions["review_reason"] = transactions["iqr_amount_alert"].map(
        {1: "Amount is above the IQR limit", 0: "No amount alert"}
    )
    return transactions, upper_limit


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run `python src/generate_data.py` first.")

    transactions = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    transactions, upper_limit = add_iqr_alerts(transactions)
    transactions.to_csv(OUTPUT_FILE, index=False)

    alerts = transactions["iqr_amount_alert"] == 1
    known_high_risk = transactions["risk_flag"] == 1
    found_examples = (alerts & known_high_risk).sum()

    print(f"IQR upper limit: INR {upper_limit:,.2f}")
    print(f"Transactions in review queue: {alerts.sum():,}")
    print(f"Known high-value examples found: {found_examples:,} out of {known_high_risk.sum():,}")
    print(f"Saved file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

