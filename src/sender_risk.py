"""Sender behavioral risk intelligence for UPI transactions.

Builds per-sender baselines, flags amounts that are extreme relative to a
sender's own history, ranks senders by composite exposure, and surfaces
high fan-in receivers that look like collection mules in the synthetic demo.

This layer is additive: it does not rewrite the fixed multi-rule composite
score in ``analyze_data.evaluate_rules``. Analysts can use it as a second
pass after the explainable rules fire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sender_risk_profiles.csv"

# How extreme a single amount must be vs the sender's own IQR fence.
SENDER_AMOUNT_IQR_MULTIPLIER = 1.5
# Minimum history before a personal baseline is trusted.
MIN_TXNS_FOR_BASELINE = 3
# Receivers with at least this many distinct senders are high fan-in.
HIGH_FAN_IN_SENDERS = 8


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"transactions are missing required columns: {sorted(missing)}")


def build_sender_profiles(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate behavioral baselines for every sender in the frame."""
    _require_columns(
        transactions,
        {"sender_id", "amount", "timestamp", "receiver_id", "upi_channel"},
    )
    if transactions.empty:
        return pd.DataFrame(
            columns=[
                "sender_id",
                "txn_count",
                "total_volume",
                "median_amount",
                "mean_amount",
                "amount_iqr",
                "personal_upper_fence",
                "unique_receivers",
                "receiver_concentration",
                "night_share",
                "collect_request_share",
                "first_seen",
                "last_seen",
            ]
        )

    frame = transactions.copy()
    if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame = frame.dropna(subset=["sender_id", "amount", "timestamp"])

    hour = frame["timestamp"].dt.hour
    frame["_night"] = ((hour >= 1) & (hour < 5)).astype(int)
    frame["_collect"] = (frame["upi_channel"].astype("string").str.strip() == "Collect Request").astype(int)

    grouped = frame.groupby("sender_id", sort=False)
    profiles = grouped.agg(
        txn_count=("amount", "size"),
        total_volume=("amount", "sum"),
        median_amount=("amount", "median"),
        mean_amount=("amount", "mean"),
        amount_q1=("amount", lambda s: float(s.quantile(0.25))),
        amount_q3=("amount", lambda s: float(s.quantile(0.75))),
        unique_receivers=("receiver_id", "nunique"),
        night_share=("_night", "mean"),
        collect_request_share=("_collect", "mean"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    ).reset_index()

    profiles["amount_iqr"] = (profiles["amount_q3"] - profiles["amount_q1"]).clip(lower=0.0)
    profiles["personal_upper_fence"] = (
        profiles["amount_q3"] + SENDER_AMOUNT_IQR_MULTIPLIER * profiles["amount_iqr"]
    )
    # Share of volume going to the single top receiver (1.0 = all money to one party).
    top_receiver_volume = (
        frame.groupby(["sender_id", "receiver_id"], sort=False)["amount"]
        .sum()
        .groupby("sender_id")
        .max()
    )
    profiles["receiver_concentration"] = (
        profiles["sender_id"].map(top_receiver_volume) / profiles["total_volume"]
    ).fillna(0.0)

    profiles = profiles.drop(columns=["amount_q1", "amount_q3"])
    for column in (
        "total_volume",
        "median_amount",
        "mean_amount",
        "amount_iqr",
        "personal_upper_fence",
        "receiver_concentration",
        "night_share",
        "collect_request_share",
    ):
        profiles[column] = profiles[column].round(4)

    return profiles.sort_values(["txn_count", "total_volume"], ascending=False).reset_index(drop=True)


def annotate_sender_amount_anomalies(
    transactions: pd.DataFrame,
    profiles: pd.DataFrame | None = None,
    *,
    min_history: int = MIN_TXNS_FOR_BASELINE,
) -> pd.DataFrame:
    """Flag rows whose amount exceeds the sender's personal Tukey fence.

    Senders with fewer than ``min_history`` prior+current observations are not
    scored on a personal baseline (flag stays 0) because the fence is unstable.
    """
    _require_columns(transactions, {"sender_id", "amount"})
    if min_history < 1:
        raise ValueError("min_history must be at least 1")

    frame = transactions.copy()
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    if profiles is None:
        profiles = build_sender_profiles(frame if "timestamp" in frame.columns else transactions)

    lookup = profiles.set_index("sender_id")[
        ["txn_count", "personal_upper_fence", "median_amount"]
    ]
    joined = frame.join(lookup, on="sender_id", how="left")
    joined["txn_count"] = joined["txn_count"].fillna(0).astype(int)
    joined["personal_upper_fence"] = pd.to_numeric(joined["personal_upper_fence"], errors="coerce")
    joined["median_amount"] = pd.to_numeric(joined["median_amount"], errors="coerce")

    eligible = joined["txn_count"] >= min_history
    above_fence = joined["amount"] > joined["personal_upper_fence"]
    joined["sender_amount_anomaly"] = (eligible & above_fence.fillna(False)).astype(int)
    # Ratio vs personal median helps analysts size the deviation quickly.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joined["amount"] / joined["median_amount"].replace(0, np.nan)
    joined["amount_vs_sender_median"] = ratio.replace([np.inf, -np.inf], np.nan).round(2)
    joined.loc[~eligible, "amount_vs_sender_median"] = np.nan
    return joined


def detect_high_fan_in_receivers(
    transactions: pd.DataFrame,
    *,
    min_unique_senders: int = HIGH_FAN_IN_SENDERS,
) -> pd.DataFrame:
    """Rank receivers by distinct-sender fan-in (mule / collection pattern)."""
    _require_columns(transactions, {"receiver_id", "sender_id", "amount"})
    if min_unique_senders < 1:
        raise ValueError("min_unique_senders must be at least 1")
    if transactions.empty:
        return pd.DataFrame(
            columns=[
                "receiver_id",
                "unique_senders",
                "txn_count",
                "total_volume",
                "high_fan_in",
            ]
        )

    frame = transactions.copy()
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    receivers = (
        frame.groupby("receiver_id", sort=False)
        .agg(
            unique_senders=("sender_id", "nunique"),
            txn_count=("sender_id", "size"),
            total_volume=("amount", "sum"),
        )
        .reset_index()
    )
    receivers["high_fan_in"] = (receivers["unique_senders"] >= min_unique_senders).astype(int)
    receivers["total_volume"] = receivers["total_volume"].round(2)
    return receivers.sort_values(
        ["high_fan_in", "unique_senders", "total_volume"], ascending=False
    ).reset_index(drop=True)


def rank_sender_risk(
    scored_transactions: pd.DataFrame,
    profiles: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine rule-queue hits with behavioral baselines into a sender leaderboard.

    Expected scored columns: ``sender_id``, ``risk_score``, ``in_review_queue``,
    and preferably ``amount``. When rule columns are absent the behavioral half
    of the score still runs.
    """
    _require_columns(scored_transactions, {"sender_id"})
    if profiles is None:
        profiles = build_sender_profiles(scored_transactions)

    frame = scored_transactions.copy()
    if "risk_score" not in frame.columns:
        frame["risk_score"] = 0
    if "in_review_queue" not in frame.columns:
        frame["in_review_queue"] = 0
    if "amount" not in frame.columns:
        frame["amount"] = 0.0
    if "sender_amount_anomaly" not in frame.columns:
        frame = annotate_sender_amount_anomalies(frame, profiles)

    agg = (
        frame.groupby("sender_id", sort=False)
        .agg(
            txn_count=("sender_id", "size"),
            mean_rule_score=("risk_score", "mean"),
            max_rule_score=("risk_score", "max"),
            review_queue_hits=("in_review_queue", "sum"),
            sender_anomaly_hits=("sender_amount_anomaly", "sum"),
            total_volume=("amount", "sum"),
        )
        .reset_index()
    )

    merged = profiles.merge(agg, on="sender_id", how="left", suffixes=("", "_agg"))
    # Prefer profile txn_count; fall back to the scored aggregate when needed.
    if "txn_count_agg" in merged.columns:
        merged["txn_count"] = merged["txn_count"].fillna(merged["txn_count_agg"])
        merged = merged.drop(columns=["txn_count_agg"])
    for column in (
        "mean_rule_score",
        "max_rule_score",
        "review_queue_hits",
        "sender_anomaly_hits",
        "total_volume",
    ):
        if column not in merged.columns:
            merged[column] = 0
        merged[column] = merged[column].fillna(0)

    # Transparent weighted blend on a 0-100 scale for ranking only.
    queue_rate = np.where(
        merged["txn_count"] > 0,
        merged["review_queue_hits"] / merged["txn_count"],
        0.0,
    )
    anomaly_rate = np.where(
        merged["txn_count"] > 0,
        merged["sender_anomaly_hits"] / merged["txn_count"],
        0.0,
    )
    merged["sender_risk_index"] = (
        0.45 * merged["mean_rule_score"].clip(0, 100)
        + 0.25 * (100 * queue_rate)
        + 0.20 * (100 * anomaly_rate)
        + 0.10 * (100 * merged["night_share"].clip(0, 1))
    ).round(2)

    def _tier(score: float) -> str:
        if score >= 60:
            return "High"
        if score >= 30:
            return "Medium"
        return "Low"

    merged["sender_risk_tier"] = merged["sender_risk_index"].map(_tier)
    merged["review_queue_hits"] = merged["review_queue_hits"].astype(int)
    merged["sender_anomaly_hits"] = merged["sender_anomaly_hits"].astype(int)
    return merged.sort_values("sender_risk_index", ascending=False).reset_index(drop=True)


def build_sender_risk_export(
    scored_transactions: pd.DataFrame,
    *,
    min_history: int = MIN_TXNS_FOR_BASELINE,
    min_unique_senders: int = HIGH_FAN_IN_SENDERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (annotated transactions, sender leaderboard, high fan-in receivers)."""
    profiles = build_sender_profiles(scored_transactions)
    annotated = annotate_sender_amount_anomalies(
        scored_transactions, profiles, min_history=min_history
    )
    leaderboard = rank_sender_risk(annotated, profiles)
    receivers = detect_high_fan_in_receivers(
        scored_transactions, min_unique_senders=min_unique_senders
    )
    return annotated, leaderboard, receivers


def main() -> None:
    """CLI: rebuild sender profiles from the enriched scored transactions file."""
    from src.analyze_data import OUTPUT_FILE  # local import keeps module light

    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {OUTPUT_FILE}. Run `python src/analyze_data.py` first."
        )
    scored = pd.read_csv(OUTPUT_FILE, parse_dates=["timestamp"])
    _, leaderboard, receivers = build_sender_risk_export(scored)
    DEFAULT_OUTPUT.parent.mkdir(exist_ok=True)
    leaderboard.to_csv(DEFAULT_OUTPUT, index=False)
    receivers_path = PROJECT_ROOT / "data" / "receiver_fan_in.csv"
    receivers.to_csv(receivers_path, index=False)
    high = int((leaderboard["sender_risk_tier"] == "High").sum())
    print("=== Sender Behavioral Risk Intelligence ===")
    print(f"Senders profiled: {len(leaderboard):,}")
    print(f"High-tier senders: {high:,}")
    print(f"High fan-in receivers: {int(receivers['high_fan_in'].sum()):,}")
    print(f"Saved sender leaderboard: {DEFAULT_OUTPUT}")
    print(f"Saved receiver fan-in: {receivers_path}")


if __name__ == "__main__":
    main()
