"""A simple dashboard for the synthetic UPI transaction review queue."""

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "transactions_with_iqr_flags.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE, parse_dates=["timestamp"])


st.set_page_config(page_title="UPI Review Queue", layout="wide")
st.title("UPI Transaction Risk Review")
st.caption("Student project - all data is synthetic.")

if not DATA_FILE.exists():
    st.error("Run the two scripts in src before opening the dashboard.")
    st.stop()

data = load_data()
alerts = data[data["iqr_amount_alert"] == 1]
known_high_risk = data[data["risk_flag"] == 1]
known_examples_found = ((data["iqr_amount_alert"] == 1) & (data["risk_flag"] == 1)).sum()

first, second, third = st.columns(3)
first.metric("Transactions", f"{len(data):,}")
second.metric("Review queue", f"{len(alerts):,}")
third.metric("Known high-value examples found", f"{known_examples_found:,} / {len(known_high_risk):,}")

st.info("The IQR rule only checks transaction amount. An alert means a human should review it, not that it is fraud.")

st.subheader("Transactions each day")
daily_transactions = data.set_index("timestamp").resample("D").size()
st.line_chart(daily_transactions)

st.subheader("Payments for manual review")
columns_to_show = ["transaction_id", "timestamp", "sender_id", "receiver_id", "amount", "merchant_category", "review_reason"]
st.dataframe(alerts.sort_values("amount", ascending=False)[columns_to_show].head(50), width="stretch")
