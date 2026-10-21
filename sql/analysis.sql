-- UPI Transaction Risk Analytics SQL Queries
-- Designed for SQLite / DuckDB / PostgreSQL execution

-- 1. Merchant Category Volume & Total Spend
SELECT 
    merchant_category, 
    COUNT(*) AS total_transactions, 
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_ticket_size
FROM upi_transactions
GROUP BY merchant_category
ORDER BY total_amount DESC;

-- 2. Hourly UPI Traffic & Diurnal Cycle Analysis
SELECT 
    strftime('%H', timestamp) AS hour_of_day, 
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS avg_amount,
    SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) AS risk_flagged_count
FROM upi_transactions
GROUP BY hour_of_day
ORDER BY hour_of_day ASC;

-- 3. High-Risk Payments by City
SELECT 
    location, 
    COUNT(*) AS total_txns,
    SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) AS high_risk_txns,
    ROUND(100.0 * SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS risk_percentage
FROM upi_transactions
GROUP BY location
ORDER BY high_risk_txns DESC;

-- 4. Channel Vulnerability Breakdown (App vs QR vs Collect Request)
SELECT 
    upi_channel,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS total_volume,
    SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) AS flagged_count,
    ROUND(100.0 * SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS incident_rate_pct
FROM upi_transactions
GROUP BY upi_channel
ORDER BY incident_rate_pct DESC;

-- 5. Sender Accounts with Rapid Consecutive Transactions (Velocity Outliers)
-- SQLite window functions identify senders with at least three transactions
-- inside a rolling 15-minute window instead of only counting daily volume.
WITH ordered_transactions AS (
    SELECT
        sender_id,
        transaction_id,
        timestamp,
        amount,
        COUNT(*) OVER (
            PARTITION BY sender_id
            ORDER BY unixepoch(timestamp)
            RANGE BETWEEN 900 PRECEDING AND CURRENT ROW
        ) AS transactions_in_15m
    FROM upi_transactions
)
SELECT 
    sender_id,
    MAX(transactions_in_15m) AS peak_transactions_in_15m,
    COUNT(*) AS triggering_transactions,
    ROUND(SUM(amount), 2) AS triggering_amount,
    MIN(timestamp) AS first_txn,
    MAX(timestamp) AS last_txn
FROM ordered_transactions
WHERE transactions_in_15m >= 3
GROUP BY sender_id
ORDER BY peak_transactions_in_15m DESC, triggering_transactions DESC
LIMIT 10;
