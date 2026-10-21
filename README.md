# UPI Transaction Risk & Fraud Analytics

Synthetic UPI payments, a rule-based risk scorer, an Isolation Forest benchmark, and a Streamlit dashboard for triage and exploration. All transaction data is fake — built for demos and coursework, not production fraud ops.

## Repo layout

```
upi-fraud-risk-analytics/
├── app/dashboard.py
├── data/
├── sql/analysis.sql
├── src/
│   ├── generate_data.py
│   ├── analyze_data.py
│   └── sql_queries.py
├── tests/
├── .github/workflows/ci.yml
└── requirements.txt
```

## Quick start

```powershell
pip install -r requirements.txt
python src/generate_data.py
python src/analyze_data.py
streamlit run app/dashboard.py
```

`analyze_data.py` writes `data/transactions_with_iqr_flags.csv`. The dashboard loads that file and **re-runs the rule engine** when you change the IQR multiplier, so queue size, tiers, and explainability stay in sync with the slider. You do not need to re-run `analyze_data.py` just to try different cutoffs.

Tests (also run in CI on push/PR):

```powershell
python -m unittest discover -s tests -v
```

## Risk rules

| Rule | Condition | Points |
|------|-----------|--------|
| Amount IQR | Above Tukey upper fence (Q3 + k×IQR) | 40 |
| Velocity | ≥3 txns by same sender within 15 minutes | 30 |
| Collect request | Collect Request and amount ≥ ₹5,000 | 20 |
| Odd hours | 01:00–05:00 and amount > ₹1,500 | 10 |

Scores sum to a 0–100 composite. Tiers: Low (0–29), Medium (30–59), High (60+). Review queue uses score ≥ 30.

Rules are fully explainable (fixed reasons per alert). Isolation Forest is an optional second pass on amount, time, velocity, channel, device, and category frequency — useful for comparison, not as the primary audit trail.

## Dashboard

- **Review queue** — search, filter, persistent analyst actions (`data/analyst_actions.sqlite3`, gitignored).
- **Analytics** — hourly patterns, channels, amounts (IQR line matches the slider), cities.
- **ML benchmark** — rules vs Isolation Forest vs synthetic `risk_flag`.
- **SQL lab** — queries loaded from `sql/analysis.sql`; read-only `SELECT`/`WITH` only.
- **Upload CSV** — same engine as the main dataset. Required: `timestamp`, `amount`, `sender_id`, `receiver_id`, `upi_channel`. Add `merchant_category` and `device_type` for ML columns.

Upload validation checks required columns, positive amounts, parseable timestamps, duplicate IDs, empty sender/receiver IDs, and supported UPI channels.

Velocity bursts in the generator are explicit three-transaction windows so the synthetic benchmark lines up with the detector.
