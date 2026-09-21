"""Run a simple IQR-based amount review rule on the generated data."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "upi_transactions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "transactions_with_iqr_flags.csv"


def add_iqr_flag(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Flag amounts above the usual range using the Interquartile Range rule."""
    first_quartile = data["amount"].quantile(0.25)
    third_quartile = data["amount"].quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    upper_limit = third_quartile + 1.5 * interquartile_range

    data = data.copy()
    data["iqr_amount_flag"] = (data["amount"] > upper_limit).astype(int)
    return data, upper_limit


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run `python src/generate_data.py` before analysing the data.")

    transactions = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    transactions, upper_limit = add_iqr_flag(transactions)
    transactions.to_csv(OUTPUT_FILE, index=False)

    flagged_transactions = transactions["iqr_amount_flag"].sum()
    print(f"IQR upper limit: INR {upper_limit:,.2f}")
    print(f"Transactions flagged for manual review: {flagged_transactions:,}")
    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

