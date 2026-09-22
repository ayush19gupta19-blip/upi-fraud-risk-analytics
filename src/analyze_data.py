"""Multi-factor risk scoring engine and machine learning anomaly detection for UPI transactions."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "upi_transactions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "transactions_with_iqr_flags.csv"


def calculate_iqr_limit(amounts: pd.Series, multiplier: float = 1.5) -> float:
    """Calculate the upper outlier cutoff using Tukey's IQR fence method."""
    if amounts.empty:
        return 0.0
    q1 = amounts.quantile(0.25)
    q3 = amounts.quantile(0.75)
    iqr = q3 - q1
    return float(q3 + multiplier * iqr)


def compute_rolling_velocity(df: pd.DataFrame, window_mins: int = 15) -> pd.Series:
    """Compute rolling transaction count per sender within a sliding time window (vectorized)."""
    window_ns = np.timedelta64(window_mins, "m")
    rolling_counts = np.ones(len(df), dtype=int)

    # df is guaranteed to be sorted by timestamp
    for _, indices in df.groupby("sender_id").groups.items():
        if len(indices) <= 1:
            continue
        sub_indices = np.array(indices)
        times = df.loc[sub_indices, "timestamp"].values
        left_bounds = np.searchsorted(times, times - window_ns, side="left")
        counts = np.arange(len(times)) - left_bounds + 1
        rolling_counts[sub_indices] = counts

    return pd.Series(rolling_counts, index=df.index, dtype=int)


def evaluate_rules(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    velocity_window_mins: int = 15,
    velocity_threshold: int = 3,
) -> tuple[pd.DataFrame, float]:
    """Evaluate explainable multi-factor risk rules on UPI transactions."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Amount IQR Rule
    upper_limit = calculate_iqr_limit(df["amount"], multiplier=iqr_multiplier)
    df["iqr_amount_alert"] = (df["amount"] > upper_limit).astype(int)

    # 2. Rolling Velocity Rule (rapid transactions by same sender)
    df["rolling_tx_count"] = compute_rolling_velocity(df, window_mins=velocity_window_mins)
    df["velocity_alert"] = (df["rolling_tx_count"] >= velocity_threshold).astype(int)

    # 3. Odd-Hours Rule (01:00 to 05:00 AM with elevated transaction amount)
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["odd_hours_alert"] = (
        (df["hour_of_day"] >= 1) & (df["hour_of_day"] < 5) & (df["amount"] > 1_500)
    ).astype(int)

    # 4. Collect Request Scam Rule (high amount requested via Collect Request)
    df["collect_scam_alert"] = (
        (df["upi_channel"] == "Collect Request") & (df["amount"] >= 5_000)
    ).astype(int)

    # 5. Composite Risk Score (0 to 100)
    # Amount IQR: 40 pts, Velocity: 30 pts, Collect Scam: 20 pts, Odd Hours: 10 pts
    df["risk_score"] = (
        df["iqr_amount_alert"] * 40
        + df["velocity_alert"] * 30
        + df["collect_scam_alert"] * 20
        + df["odd_hours_alert"] * 10
    ).clip(0, 100)

    # Categorical Risk Level
    conditions = [
        df["risk_score"] >= 60,
        df["risk_score"] >= 30,
    ]
    choices = ["High", "Medium"]
    df["risk_level"] = np.select(conditions, choices, default="Low")

    # Generate Explainable Review Reasons
    def build_review_reason(row: pd.Series) -> str:
        reasons = []
        if row["iqr_amount_alert"] == 1:
            reasons.append(f"Amount > INR {upper_limit:,.0f} (IQR)")
        if row["velocity_alert"] == 1:
            reasons.append(f"Velocity spike ({row['rolling_tx_count']} txns in {velocity_window_mins}m)")
        if row["collect_scam_alert"] == 1:
            reasons.append("High-value Collect Request")
        if row["odd_hours_alert"] == 1:
            reasons.append(f"Elevated amount at {row['hour_of_day']:02d}:00 hrs")

        return "; ".join(reasons) if reasons else "No risk flags triggered"

    df["review_reason"] = df.apply(build_review_reason, axis=1)
    df["in_review_queue"] = (df["risk_score"] >= 30).astype(int)

    return df, upper_limit


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.035, seed: int = 42) -> pd.DataFrame:
    """Train an unsupervised Isolation Forest model to benchmark against rule-based scoring."""
    df = df.copy()

    # Feature engineering for ML
    features = pd.DataFrame(index=df.index)
    features["amount_log"] = np.log1p(df["amount"])
    features["rolling_tx_count"] = df["rolling_tx_count"]
    features["hour_of_day"] = df["hour_of_day"]
    features["is_collect_request"] = (df["upi_channel"] == "Collect Request").astype(int)
    features["is_ios"] = (df["device_type"] == "iOS").astype(int)

    # Frequency encoding for category
    cat_freq = df["merchant_category"].value_counts(normalize=True)
    features["category_freq"] = df["merchant_category"].map(cat_freq)

    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    raw_preds = iso.fit_predict(features)
    decision_func = iso.decision_function(features)

    df["ml_prediction"] = (raw_preds == -1).astype(int)
    # Scale score to 0 - 100 where higher means more anomalous
    denom = decision_func.max() - decision_func.min()
    if denom > 0:
        normalized_score = 100 * (1 - (decision_func - decision_func.min()) / denom)
    else:
        normalized_score = np.zeros(len(df))
    df["ml_anomaly_score"] = np.round(normalized_score, 1)

    return df


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def add_iqr_alerts(transactions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Backwards-compatible wrapper adding multi-factor rules and ML model."""
    df, upper_limit = evaluate_rules(transactions)
    df = train_isolation_forest(df)
    return df, upper_limit


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run `python src/generate_data.py` first.")

    transactions = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
    enriched_df, upper_limit = add_iqr_alerts(transactions)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    enriched_df.to_csv(OUTPUT_FILE, index=False)

    print(f"=== UPI Transaction Risk Analytics Engine ===")
    print(f"IQR Upper Threshold: INR {upper_limit:,.2f}")
    print(f"Total Transactions: {len(enriched_df):,}")
    print(f"Transactions in Review Queue: {enriched_df['in_review_queue'].sum():,}")

    rule_metrics = calculate_metrics(enriched_df["risk_flag"], enriched_df["in_review_queue"])
    ml_metrics = calculate_metrics(enriched_df["risk_flag"], enriched_df["ml_prediction"])

    print("\n--- Rule-Based Engine Performance vs Synthetic Injected Ground Truth ---")
    print(f"True Positives:  {rule_metrics['tp']:,} / {enriched_df['risk_flag'].sum():,}")
    print(f"False Positives: {rule_metrics['fp']:,}")
    print(f"Precision:       {rule_metrics['precision']:.2%}")
    print(f"Recall:          {rule_metrics['recall']:.2%}")
    print(f"F1 Score:        {rule_metrics['f1']:.2%}")

    print("\n--- Isolation Forest (ML) Performance ---")
    print(f"True Positives:  {ml_metrics['tp']:,} / {enriched_df['risk_flag'].sum():,}")
    print(f"False Positives: {ml_metrics['fp']:,}")
    print(f"Precision:       {ml_metrics['precision']:.2%}")
    print(f"Recall:          {ml_metrics['recall']:.2%}")
    print(f"F1 Score:        {ml_metrics['f1']:.2%}")

    both = ((enriched_df["in_review_queue"] == 1) & (enriched_df["ml_prediction"] == 1)).sum()
    print(f"\nAgreed Detections (Both Rules & ML): {both:,}")
    print(f"Saved enriched data: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
