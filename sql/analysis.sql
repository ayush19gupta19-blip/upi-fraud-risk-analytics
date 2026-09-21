-- Import data/upi_transactions.csv into SQLite as upi_transactions.

-- 1. Which merchant categories receive the largest total payment value?
SELECT merchant_category, COUNT(*) AS transaction_count, ROUND(SUM(amount), 2) AS total_amount
FROM upi_transactions
GROUP BY merchant_category
ORDER BY total_amount DESC;

-- 2. Which hours have the most UPI activity?
SELECT strftime('%H', timestamp) AS hour_of_day, COUNT(*) AS transaction_count
FROM upi_transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- 3. Which locations have the most synthetic high-value examples?
SELECT location, COUNT(*) AS high_value_examples
FROM upi_transactions
WHERE risk_flag = 1
GROUP BY location
ORDER BY high_value_examples DESC;

