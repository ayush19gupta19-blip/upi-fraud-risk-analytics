# UPI Fraud Risk Analytics

A student portfolio project that simulates UPI transactions and highlights transactions that deserve manual review.

> This project uses fully synthetic data. It is an educational risk-scoring demo, not a real fraud-prevention system.

## What this project demonstrates

- Generating a realistic-looking transaction dataset
- Cleaning data and exploring it with Python and SQL
- Applying an easy-to-explain IQR outlier rule
- Building a small dashboard for a fraud-review team

## Project structure

```
app/                 Streamlit dashboard
data/                Generated CSV files (not real payment data)
src/                 Python scripts
sql/                 SQL questions and answers
tests/               Simple checks for the generator
```

## Dataset fields

| Field | Meaning |
| --- | --- |
| `transaction_id` | Unique ID for each synthetic transaction |
| `timestamp` | Date and time of the transaction |
| `sender_id` / `receiver_id` | Simulated customer and recipient IDs |
| `amount` | UPI transaction value in INR |
| `merchant_category` | Category used for the payment |
| `location` | Simulated city label |
| `device_type` | Device used for the transaction |
| `upi_channel` | Channel through which the transaction was made |
| `risk_flag` | Synthetic label used only to evaluate this demo |
| `risk_scenario` | The pattern deliberately injected into the synthetic data |

## Run the project

Create a virtual environment, install the packages, then run the commands below from the project folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/generate_data.py
python src/analyze_data.py
streamlit run app/dashboard.py
```

## What the first version does

The generator creates 10,000 transactions. Most are normal transactions. A small number contain one deliberately injected review pattern: an unusually high amount, a late-night payment, or a rapid repeat payment.

`analyze_data.py` uses the IQR rule to flag unusually large amounts. The dashboard shows the transaction trend, amount distribution, and the review queue.

## Limitations

- Fraud labels are synthetic, so the results do not prove real-world accuracy.
- IQR looks only at transaction amount; it cannot find every suspicious pattern.
- A real payment-risk system needs privacy controls, human review, monitoring, and access to trustworthy historical data.

## Interview explanation

“I built a UPI transaction-risk analytics demo using simulated data because real payment fraud data is private. I first explored the data with SQL and Python, then used the IQR method to find unusually high transactions. The dashboard helps a review team see which alerts need attention. I clearly documented that the labels and results are synthetic.”

