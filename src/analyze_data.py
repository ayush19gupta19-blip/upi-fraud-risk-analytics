"""Multi-factor risk scoring engine and machine learning anomaly detection for UPI transactions."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "upi_transactions.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "transactions_with_iqr_flags.csv"

# Composite risk rule weights (points out of 100). Single source of truth for
# both scoring and per-transaction explainability.
RULE_WEIGHTS: dict[str, int] = {
    "iqr_amount_alert": 40,   # Amount outlier vs Tukey IQR fence
    "velocity_alert": 30,     # Rapid-fire transactions by the same sender
    "collect_scam_alert": 20, # High-value Collect Request (phishing vector)
    "odd_hours_alert": 10,    # Elevated amount inside the 01:00-05:00 window
}

# Composite score thresholds for each risk tier
RISK_TIER_THRESHOLDS: dict[str, int] = {"Medium": 30, "High": 60}

# Columns a user-supplied CSV must contain to be scoreable
REQUIRED_UPLOAD_COLUMNS = ["timestamp", "amount", "sender_id", "receiver_id", "upi_channel"]

# Extra columns needed for the Isolation Forest benchmark (optional on upload)
ML_FEATURE_COLUMNS = ["merchant_category", "device_type"]


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

    # 5. Composite Risk Score (0 to 100) from the weighted rule set
    df["risk_score"] = sum(
        df[column] * weight for column, weight in RULE_WEIGHTS.items()
    ).clip(0, 100)

    # Categorical Risk Level
    conditions = [
        df["risk_score"] >= RISK_TIER_THRESHOLDS["High"],
        df["risk_score"] >= RISK_TIER_THRESHOLDS["Medium"],
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


def explain_transaction(
    row: pd.Series,
    iqr_upper_limit: float | None = None,
    velocity_window_mins: int = 15,
    velocity_threshold: int = 3,
) -> dict:
    """Break a scored transaction's composite risk score into per-rule contributions.

    Returns an audit-ready dict: the score and tier, a per-rule breakdown
    (fired / points / plain-English detail), and the gap to the next tier.
    """
    amount = float(row.get("amount", 0.0) or 0.0)
    rolling_count = int(row.get("rolling_tx_count", 1) or 1)
    hour = int(row.get("hour_of_day", 0) or 0)

    labels = {
        "iqr_amount_alert": "Amount IQR Outlier",
        "velocity_alert": "Rolling Velocity Spike",
        "collect_scam_alert": "Collect Request Scam",
        "odd_hours_alert": "Odd-Hours Activity",
    }
    cutoff_text = f" of INR {iqr_upper_limit:,.2f}" if iqr_upper_limit is not None else ""
    details = {
        "iqr_amount_alert": f"Amount INR {amount:,.2f} exceeds the Tukey IQR cutoff{cutoff_text}",
        "velocity_alert": (
            f"{rolling_count} transactions by this sender within a {velocity_window_mins}-minute "
            f"window (fires at >= {velocity_threshold})"
        ),
        "collect_scam_alert": (
            f"Collect Request of INR {amount:,.2f} is at or above the INR 5,000 scam threshold"
        ),
        "odd_hours_alert": (
            f"INR {amount:,.2f} moved at {hour:02d}:00 (01:00-05:00 window with amount > INR 1,500)"
        ),
    }

    rules = []
    for column, weight in RULE_WEIGHTS.items():
        fired = int(row.get(column, 0) or 0) == 1
        rules.append(
            {
                "label": labels[column],
                "column": column,
                "fired": fired,
                "points": weight if fired else 0,
                "max_points": weight,
                "detail": details[column]
                if fired
                else f"Not triggered (worth {weight} pts if it fires)",
            }
        )
    # Fired rules first, then by descending weight
    rules.sort(key=lambda rule: (not rule["fired"], -rule["max_points"]))

    score = int(row.get("risk_score", sum(rule["points"] for rule in rules)))

    next_tier = None
    for tier in ("Medium", "High"):
        threshold = RISK_TIER_THRESHOLDS[tier]
        if score < threshold:
            next_tier = {"tier": tier, "threshold": threshold, "points_needed": threshold - score}
            break

    risk_level = row.get("risk_level")
    if risk_level is None:
        risk_level = "High" if score >= RISK_TIER_THRESHOLDS["High"] else "Medium" if score >= RISK_TIER_THRESHOLDS["Medium"] else "Low"

    return {
        "transaction_id": row.get("transaction_id"),
        "risk_score": score,
        "risk_level": str(risk_level),
        "fired_rules": sum(1 for rule in rules if rule["fired"]),
        "total_rules": len(rules),
        "rules": rules,
        "next_tier": next_tier,
    }


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


def validate_transaction_upload(df: pd.DataFrame) -> list[str]:
    """Validate a user-supplied transaction frame.

    Returns a list of human-readable problems; an empty list means the data
    can be scored by the rule engine.
    """
    problems: list[str] = []

    missing = [column for column in REQUIRED_UPLOAD_COLUMNS if column not in df.columns]
    if missing:
        found = ", ".join(map(str, df.columns)) if len(df.columns) else "none"
        problems.append(f"Missing required column(s): {', '.join(missing)}. Found: {found}.")

    if len(df) == 0:
        problems.append("The file contains no transaction rows.")
        return problems

    if "amount" in df.columns:
        amounts = pd.to_numeric(df["amount"], errors="coerce")
        if amounts.isna().any():
            problems.append(f"'amount' has {int(amounts.isna().sum())} value(s) that are not numeric.")
        elif (amounts <= 0).any():
            problems.append("'amount' must be strictly positive.")

    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
        if parsed.isna().any():
            problems.append(
                f"'timestamp' has {int(parsed.isna().sum())} value(s) that could not be parsed as dates."
            )

    return problems


def score_uploaded_transactions(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, float]:
    """Score a user-supplied transaction frame with the explainable rule engine.

    Runs the Isolation Forest benchmark as well when the optional feature
    columns (merchant_category, device_type) are present.
    Raises ValueError listing every validation problem when the data is unusable.
    """
    problems = validate_transaction_upload(df)
    if problems:
        raise ValueError("\n".join(problems))

    scored = df.copy()
    scored["timestamp"] = pd.to_datetime(scored["timestamp"], errors="coerce", format="mixed")
    scored["amount"] = pd.to_numeric(scored["amount"], errors="coerce")

    if "transaction_id" not in scored.columns:
        scored.insert(0, "transaction_id", [f"UPLOAD{i:05d}" for i in range(1, len(scored) + 1)])

    scored, upper_limit = evaluate_rules(scored, iqr_multiplier=iqr_multiplier)

    if all(column in scored.columns for column in ML_FEATURE_COLUMNS):
        scored = train_isolation_forest(scored)

    return scored, upper_limit


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
