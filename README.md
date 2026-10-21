# UPI Transaction Risk & Fraud Analytics

Synthetic UPI payment data, a rule-based risk scorer, an Isolation Forest benchmark, and a Streamlit dashboard for triage and exploration.

All transactions are generated for demo and learning — not real customer data.

## What's in the repo

```
upi-fraud-risk-analytics/
├── app/dashboard.py          # Streamlit UI
├── data/                     # CSV inputs and scored outputs
├── sql/analysis.sql          # Shared SQL templates (also used by the dashboard)
├── src/
│   ├── generate_data.py      # Synthetic data + injected fraud patterns
│   ├── analyze_data.py       # Scoring engine + ML benchmark
│   └── sql_queries.py        # Parser for analysis.sql
├── tests/
└── requirements.txt
```

## Setup

```powershell
pip install -r requirements.txt
```

Generate data:

```powershell
python src/generate_data.py
```

Score offline and write `data/transactions_with_iqr_flags.csv` (optional if you only use the dashboard):

```powershell
python src/analyze_data.py
```

Run the dashboard:

```powershell
streamlit run app/dashboard.py
```

The app reads `data/upi_transactions.csv` and **re-scores on load**. Moving the IQR multiplier in the sidebar updates risk scores, the review queue, charts, and explainability — no need to re-run `analyze_data.py` for that.

Tests:

```powershell
python -m unittest discover -s tests -v
```

CI runs the same unittest command on push (see `.github/workflows/tests.yml`).

## Risk rules

| Rule | Condition | Points |
|------|-----------|--------|
| Amount IQR | Amount above Tukey upper fence (Q3 + k×IQR) | 40 |
| Velocity | ≥3 txns by same sender within 15 minutes | 30 |
| Collect request | Collect Request and amount ≥ ₹5,000 | 20 |
| Odd hours | 01:00–05:00 and amount > ₹1,500 | 10 |

Composite score is the sum of triggered rules (capped at 100).

| Tier | Score |
|------|-------|
| Low | 0–29 |
| Medium | 30–59 |
| High | 60–100 |

Review queue: score ≥ 30 (`in_review_queue`).

## Rules vs Isolation Forest

The rule engine is fully explainable — every alert has a fixed reason string. Isolation Forest adds a second pass on amount, time, velocity, channel, device, and category frequency. The dashboard compares precision/recall against the synthetic `risk_flag` column from data generation.

## Dashboard tabs

1. **Review queue** — filter, search, analyst actions (approve / KYC / block), per-transaction rule breakdown.
2. **Analytics** — hourly traffic, channels, amount distribution (IQR line matches the sidebar), cities.
3. **ML benchmark** — rules vs Isolation Forest on the current scored dataset.
4. **SQL lab** — queries from `sql/analysis.sql`, plus a free-form editor against an in-memory SQLite copy of the scored data.
5. **Upload CSV** — same scorer as the main dataset; uses the sidebar IQR multiplier. Include `merchant_category` and `device_type` to enable the ML columns.

Required upload columns: `timestamp`, `amount`, `sender_id`, `receiver_id`, `upi_channel`.
