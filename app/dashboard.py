"""A small dashboard for exploring the synthetic UPI review queue."""

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "transactions_with_iqr_flags.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the analysis output created by src/analyze_data.py."""
    return pd.read_csv(DATA_FILE, parse_dates=["timestamp"])


st.set_page_config(page_title="UPI Risk Analytics", layout="wide")
st.title("UPI Fraud Risk Analytics")
st.caption("Student portfolio demo - all transactions and labels are synthetic.")

if not DATA_FILE.exists():
    st.error("Run `python src/generate_data.py` and `python src/analyze_data.py` first.")
    st.stop()

data = load_data()
flagged = data[data["iqr_amount_flag"] == 1]

first_column, second_column, third_column = st.columns(3)
first_column.metric("Total transactions", f"{len(data):,}")
second_column.metric("Amount alerts", f"{len(flagged):,}")
third_column.metric("Alert rate", f"{len(flagged) / len(data):.1%}")

st.subheader("Daily transaction value")
daily_amount = data.set_index("timestamp").resample("D")["amount"].sum()
st.line_chart(daily_amount)

st.subheader("Top transactions in the manual-review queue")
review_columns = ["transaction_id", "timestamp", "sender_id", "receiver_id", "amount", "merchant_category", "risk_scenario"]
st.dataframe(flagged.sort_values("amount", ascending=False)[review_columns].head(50), use_container_width=True)

