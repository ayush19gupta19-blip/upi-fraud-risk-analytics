-- Run these queries in SQLite after importing data/upi_transactions.csv
-- They answer simple business questions before building any ML model.

-- 1. How many transactions happen in each merchant category?
SELECT merchant_category, COUNT(*) AS transaction_count, ROUND(SUM(amount), 2) AS total_amount
FROM upi_transactions
GROUP BY merchant_category
ORDER BY total_amount DESC;

-- 2. Which hours have the most transaction activity?
SELECT strftime('%H', timestamp) AS hour_of_day, COUNT(*) AS transaction_count
FROM upi_transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- 3. What amount is associated with each synthetic review scenario?
SELECT risk_scenario, COUNT(*) AS transaction_count, ROUND(AVG(amount), 2) AS average_amount
FROM upi_transactions
GROUP BY risk_scenario
ORDER BY average_amount DESC;

-- 4. Which recipients receive the largest total amount?
SELECT receiver_id, COUNT(*) AS transaction_count, ROUND(SUM(amount), 2) AS total_amount
FROM upi_transactions
GROUP BY receiver_id
ORDER BY total_amount DESC
LIMIT 10;

