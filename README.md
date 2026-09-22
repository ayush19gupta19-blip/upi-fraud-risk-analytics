# 🛡️ UPI Transaction Risk & Fraud Analytics

A modern, production-grade financial risk analytics platform that combines **explainable multi-factor heuristics** (Tukey's IQR Outlier Detection, Rolling Velocity Checks, Odd-Hours Flags, and Collect-Request Scams) with an **unsupervised Machine Learning benchmark** (Isolation Forest) on synthetic UPI payments.

> **Disclaimer:** Every transaction in this project is synthetic and generated for demonstration and research purposes. It simulates real-world payment risk dynamics and compliance review operations.

---

## ⚡ Key Highlights

* **Multi-Factor Risk Scoring Engine:** Evaluates transactions across multiple behavioral dimensions and produces a composite 0–100 risk score and tiered risk categorization (`High`, `Medium`, `Low`).
* **Unsupervised ML Benchmark:** Compares the deterministic, regulatory-friendly rule engine against an `IsolationForest` model to analyze precision, recall, and false-positive trade-offs.
* **Interactive Triage Dashboard:** Built with Streamlit and Plotly, featuring live sensitivity tuning (dynamic IQR slider), triage workflow with session-based analyst decisions (`Approve`, `Verify KYC`, `Block`), and deep visual analytics.
* **Embedded SQL Lab:** An in-memory SQLite analytics engine allowing live execution of business intelligence queries from `sql/analysis.sql`.
* **Automated Test Coverage:** Comprehensive test suite covering data generator schema integrity, multi-pattern attack injections, rule evaluation, score bounds, and metric computation.

---

## 🏗️ Architecture & Component Overview

```
upi-fraud-risk-analytics/
├── app/
│   └── dashboard.py               # Streamlit interactive operations & analytics workstation
├── data/
│   ├── upi_transactions.csv       # Raw synthetic dataset (10,000 transactions)
│   └── transactions_with_iqr_flags.csv  # Enriched dataset with rules, composite scores & ML predictions
├── sql/
│   └── analysis.sql               # SQLite business analytics & risk intelligence queries
├── src/
│   ├── generate_data.py           # Diurnal cycle modeling & multi-pattern attack injection
│   └── analyze_data.py            # Multi-factor scoring engine & Isolation Forest benchmark
├── tests/
│   ├── test_generate_data.py      # Tests for data generator & schema validation
│   └── test_analyze_data.py       # Tests for IQR fences, velocity rules & scoring bounds
├── requirements.txt               # Dependencies: pandas, numpy, streamlit, scikit-learn, plotly
└── README.md
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Generate Synthetic UPI Dataset
Models realistic diurnal payment patterns (midday/evening volume peaks vs. night troughs) and injects multi-pattern fraud typologies:
```powershell
python src/generate_data.py
```

### 3. Run Risk Scoring & ML Anomaly Detection
Computes the dynamic IQR cutoff, evaluates rolling velocity spikes, odd-hour transactions, collect scams, and fits the Isolation Forest model:
```powershell
python src/analyze_data.py
```

### 4. Launch the Interactive Dashboard
```powershell
streamlit run app/dashboard.py
```

### 5. Run Automated Tests
```powershell
python -m unittest discover tests
```

---

## 🧠 Risk Detection Heuristics

| Rule | Detection Methodology | Weight | Rationale |
| :--- | :--- | :--- | :--- |
| **Amount IQR Alert** | Amount exceeds $Q3 + 1.5 \times \text{IQR}$ | 40 pts | Catches extreme monetary anomalies outside normal retail spend. |
| **Rolling Velocity Spike** | $\ge 3$ transactions by the same sender in $< 15$ minutes | 30 pts | Identifies rapid account draining or brute-force card/UPI abuse. |
| **Collect Request Scam** | Amount $\ge \text{₹5,000}$ via `Collect Request` channel | 20 pts | Detects the common "click approve to receive cashback" phishing scam. |
| **Odd-Hours Payment** | Payment between 01:00 AM and 05:00 AM with elevated value | 10 pts | Flags off-hours transactions that deviate from the sender's normal circadian pattern. |

* **Composite Risk Score (0 - 100):** Sum of triggered rule points.
* **Risk Tiers:**
  * **High Risk (60 - 100 pts):** Immediate review / block recommended.
  * **Medium Risk (30 - 59 pts):** Automated step-up authentication (OTP/KYC).
  * **Low Risk (0 - 29 pts):** Straight-through processing.

---

## 📊 Heuristics vs. Machine Learning Benchmark

In financial fraud operations, **explainability** is a critical regulatory mandate:

* **Rule-Based Engine:** Provides **100% deterministic audit trails** with exact human-readable reasons (e.g., *"Amount > INR 1,172 (IQR); Velocity spike (3 txns in 15m)"*).
* **Isolation Forest:** Effectively flags multi-dimensional outliers without manually configured thresholds, but acts as a black box requiring post-hoc explainers (such as SHAP) for regulatory compliance.

The interactive dashboard includes a dedicated **ML vs. Rule Benchmark** tab comparing precision, recall, and detection overlap between both paradigms.

---

## 📈 Dashboard Features

1. **Operational Review Queue:** Searchable triage table with live analyst actions (`Approve`, `Verify KYC`, `Block`) updating in real-time.
2. **Dynamic Sensitivity Slider:** Adjust the IQR multiplier ($1.0\times$ to $3.5\times$) live to visualize how threshold tuning impacts review queue volume.
3. **Risk Visualizations (Plotly):** Diurnal hourly risk curve, channel vulnerability breakdown, log-scale amount distributions per merchant category, and geographic risk distribution.
4. **SQL Analytics Lab:** Run pre-built or custom queries directly against an in-memory SQLite database.
