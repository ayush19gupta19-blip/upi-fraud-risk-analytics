# UPI Transaction Risk Review

A student data-analytics project that finds unusually high-value UPI payments for manual review.

**Important:** every transaction in this project is synthetic. This is a learning demo, not a real fraud-prevention system.

## Explain it in 30 seconds

“I generated 10,000 sample UPI transactions because real payment data is private. I used the IQR method to find unusually high transaction amounts and displayed those alerts in a simple dashboard. I also measured how many deliberately injected high-value examples the rule found.”

## Read these three files

1. `src/generate_data.py` - creates the sample transactions.
2. `src/analyze_data.py` - calculates the IQR limit and creates alerts.
3. `app/dashboard.py` - shows the results in Streamlit.

## Run it

```powershell
pip install -r requirements.txt
python src/generate_data.py
python src/analyze_data.py
streamlit run app/dashboard.py
```

## What the data contains

`transaction_id`, `timestamp`, `sender_id`, `receiver_id`, `amount`, `merchant_category`, `location`, `device_type`, and `upi_channel`.

Three percent of the sample payments are deliberately made very high in value. `risk_flag` is included only to check whether the IQR rule found those known synthetic examples. A real reviewer would not see that label.

## What IQR means

IQR is a basic method for finding unusually large or small values. This project calculates an upper amount limit from the normal transactions. Any payment above that limit is placed in the review queue.

## Limitations

- Synthetic labels cannot prove real-world fraud accuracy.
- Amount alone is not enough to identify fraud.
- An alert means “review this payment,” not “this payment is fraud.”

## Next improvement

Add time-of-day or repeat-payment rules after the first version is fully understood.

