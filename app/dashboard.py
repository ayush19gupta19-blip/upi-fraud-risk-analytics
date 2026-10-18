"""Interactive UPI Fraud & Transaction Risk Analytics Platform."""

from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "transactions_with_iqr_flags.csv"
RAW_DATA_FILE = PROJECT_ROOT / "data" / "upi_transactions.csv"

# Make the repo root importable so the scoring engine can be reused directly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze_data import (
    evaluate_rules,
    explain_transaction,
    score_uploaded_transactions,
    validate_transaction_upload,
)

st.set_page_config(
    page_title="UPI Risk Guardian - Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dashboard look
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-med {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_transaction_data() -> pd.DataFrame:
    """Load enriched transaction dataset with cached execution."""
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    return df


@st.cache_resource
def get_sqlite_connection(df: pd.DataFrame) -> sqlite3.Connection:
    """Create an in-memory SQLite database populated with transaction records."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    # Convert timestamp to string for sqlite compatibility
    df_sql = df.copy()
    df_sql["timestamp"] = df_sql["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_sql.to_sql("upi_transactions", conn, if_exists="replace", index=False)
    return conn


@st.cache_resource
def get_actions_connection() -> sqlite3.Connection:
    """Open the local analyst-decision store and create its schema."""
    db_path = PROJECT_ROOT / "data" / "analyst_actions.sqlite3"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analyst_actions (
            transaction_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_analyst_actions() -> dict[str, str]:
    """Load persisted decisions for display in the review queue."""
    rows = get_actions_connection().execute(
        "SELECT transaction_id, status FROM analyst_actions"
    ).fetchall()
    return {transaction_id: status for transaction_id, status in rows}


def save_analyst_action(transaction_id: str, status: str) -> None:
    """Persist one analyst decision with an auditable UTC timestamp."""
    get_actions_connection().execute(
        """
        INSERT INTO analyst_actions(transaction_id, status, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(transaction_id) DO UPDATE SET
            status = excluded.status,
            updated_at = excluded.updated_at
        """,
        (transaction_id, status, datetime.now(timezone.utc).isoformat()),
    )
    get_actions_connection().commit()


@st.cache_data(show_spinner="Recalculating dashboard risk scores...")
def score_dashboard_data(df: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Recalculate rule decisions for the current IQR sensitivity setting."""
    ml_columns = [column for column in ("ml_prediction", "ml_anomaly_score") if column in df]
    scored, _ = evaluate_rules(
        df.drop(columns=ml_columns, errors="ignore"),
        iqr_multiplier=multiplier,
    )
    if ml_columns:
        scored = scored.merge(df[["transaction_id", *ml_columns]], on="transaction_id", how="left")
    return scored


def run_read_only_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    """Execute only one read-only SQL statement in the SQL lab."""
    normalized = query.strip().upper()
    if not normalized.startswith(("SELECT", "WITH")) or ";" in normalized.rstrip(";"):
        raise ValueError("Only one read-only SELECT/WITH query is allowed.")
    forbidden = (" INSERT ", " UPDATE ", " DELETE ", " DROP ", " ALTER ", " CREATE ", " ATTACH ", " DETACH ", " PRAGMA ")
    padded = f" {normalized} "
    if any(keyword in padded for keyword in forbidden):
        raise ValueError("The SQL lab permits read-only queries only.")
    return pd.read_sql_query(query, conn)


@st.cache_data(show_spinner="Scoring your data with the risk engine...")
def score_upload_cached(uploaded_df: pd.DataFrame, multiplier: float) -> tuple[pd.DataFrame, float]:
    """Cached wrapper so re-renders don't re-score an unchanged upload."""
    return score_uploaded_transactions(uploaded_df, iqr_multiplier=multiplier)


def render_explanation(row: pd.Series, iqr_upper_limit: float) -> None:
    """Render the per-rule score breakdown backing one transaction's risk score."""
    explanation = explain_transaction(
        row,
        iqr_upper_limit=iqr_upper_limit,
        velocity_window_mins=15,
        velocity_threshold=3,
    )

    summary_c1, summary_c2 = st.columns([3, 2])
    with summary_c1:
        st.markdown(
            f"**Composite score: {explanation['risk_score']}/100 → {explanation['risk_level']} risk tier**"
            f" · {explanation['fired_rules']} of {explanation['total_rules']} rules fired"
        )
    with summary_c2:
        if explanation["next_tier"] is not None:
            st.caption(
                f"{explanation['next_tier']['points_needed']} more points would escalate to "
                f"**{explanation['next_tier']['tier']}** (threshold {explanation['next_tier']['threshold']})."
            )
        else:
            st.caption("Already at the highest risk tier — no escalation needed.")

    breakdown = pd.DataFrame(explanation["rules"])
    fig_rules = go.Figure(
        go.Bar(
            x=breakdown["max_points"],
            y=breakdown["label"],
            orientation="h",
            marker_color=["#dc2626" if fired else "#e2e8f0" for fired in breakdown["fired"]],
            text=[f"{points} pts" for points in breakdown["points"]],
            textposition="outside",
            hovertemplate="%{y}: %{x} pts<extra></extra>",
        )
    )
    fig_rules.update_layout(
        height=230,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="Rule weight (points)", range=[0, 50]),
        showlegend=False,
    )
    st.plotly_chart(fig_rules, use_container_width=True)

    for rule in explanation["rules"]:
        icon = "🔴" if rule["fired"] else "⚪"
        st.markdown(
            f"{icon} **{rule['label']}** `{rule['points']}/{rule['max_points']} pts` — {rule['detail']}"
        )


if not DATA_FILE.exists():
    st.error("Missing enriched data file. Run `python src/generate_data.py` and `python src/analyze_data.py` first.")
    st.stop()

raw_df = load_transaction_data()
analyst_actions = load_analyst_actions()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=64)
st.sidebar.title("Filter & Tuning Controls")

# Date Filter
min_date = raw_df["timestamp"].dt.date.min()
max_date = raw_df["timestamp"].dt.date.max()
selected_dates = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Risk Level Filter
available_risk_levels = ["High", "Medium", "Low"]
selected_risk_levels = st.sidebar.multiselect(
    "Risk Tiers", options=available_risk_levels, default=["High", "Medium"]
)

# UPI Channel Filter
available_channels = list(raw_df["upi_channel"].unique())
selected_channels = st.sidebar.multiselect("UPI Channel", options=available_channels, default=available_channels)

# Merchant Category Filter
available_categories = list(raw_df["merchant_category"].unique())
selected_categories = st.sidebar.multiselect("Merchant Category", options=available_categories, default=available_categories)

# City Filter
available_cities = list(raw_df["location"].unique())
selected_cities = st.sidebar.multiselect("Cities", options=available_cities, default=available_cities)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Live IQR Sensitivity Tuning")
iqr_mult = st.sidebar.slider(
    "IQR Multiplier (Fence Sensitivity)",
    min_value=1.0,
    max_value=3.5,
    value=1.5,
    step=0.25,
    help="Default is 1.5x (standard Tukey's fence). Lower values increase review queue size (higher sensitivity).",
)

# Recalculate dynamic upper limit based on slider
q1 = raw_df["amount"].quantile(0.25)
q3 = raw_df["amount"].quantile(0.75)
dynamic_iqr_limit = q3 + iqr_mult * (q3 - q1)
st.sidebar.caption(f"Dynamic Upper Cutoff: **INR {dynamic_iqr_limit:,.2f}**")

# Re-score the complete dataset so every dashboard view reflects the selected multiplier.
scored_df = score_dashboard_data(raw_df, float(iqr_mult))

# Apply Filters
mask = (
    (scored_df["upi_channel"].isin(selected_channels))
    & (scored_df["merchant_category"].isin(selected_categories))
    & (scored_df["location"].isin(selected_cities))
)
if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    mask = mask & (scored_df["timestamp"].dt.date >= start_d) & (scored_df["timestamp"].dt.date <= end_d)

if selected_risk_levels:
    mask = mask & (scored_df["risk_level"].isin(selected_risk_levels))

filtered_df = scored_df[mask].copy()

if filtered_df.empty:
    st.warning("⚠️ No transactions match the selected filters. Please adjust your criteria in the sidebar.")
    st.stop()

# ----------------- MAIN HEADER & KPI METRICS -----------------
st.title("🛡️ UPI Transaction Risk & Fraud Analytics")
st.caption("Production-grade Risk Decisioning Engine: Explainable Multi-Rule Heuristics vs. Unsupervised Isolation Forest")

# Top KPI row
c1, c2, c3, c4, c5 = st.columns(5)
total_in_view = len(filtered_df)
high_risk_count = (filtered_df["risk_level"] == "High").sum()
queue_count = (filtered_df["in_review_queue"] == 1).sum()
avg_ticket = filtered_df["amount"].mean() if total_in_view > 0 else 0.0
total_volume = filtered_df["amount"].sum()

c1.metric("Transactions In Scope", f"{total_in_view:,}", help="Total transactions matching selected filters")
c2.metric("In Review Queue", f"{queue_count:,}", f"{(queue_count / total_in_view * 100):.1f}% queue rate" if total_in_view else "0%")
c3.metric("High-Risk Alerts", f"{high_risk_count:,}", delta_color="inverse")
c4.metric("Average Ticket", f"INR {avg_ticket:,.0f}")
c5.metric("Total Volume", f"INR {total_volume / 1e6:.2f}M")

st.markdown("---")

# ----------------- TABS LAYOUT -----------------
tab_queue, tab_analytics, tab_ml_benchmark, tab_sql, tab_upload = st.tabs([
    "🚨 Operational Review Queue",
    "📈 Risk Analytics & Patterns",
    "🤖 ML vs. Rule Benchmark",
    "🗄️ Interactive SQL Lab",
    "📤 Score Your Own CSV",
])

# ================= TAB 1: OPERATIONAL REVIEW QUEUE =================
with tab_queue:
    st.subheader("Manual Review Triage Workstation")
    st.info("Transactions prioritized by Composite Risk Score (combining Amount Outlier, Velocity Spikes, Odd Hours, and Channel Phishing risk).")

    q_col1, q_col2 = st.columns([3, 1])
    with q_col1:
        search_query = st.text_input("🔍 Search by Transaction ID or Sender ID", placeholder="e.g. TXN000123 or USER0456")
    with q_col2:
        only_in_queue = st.checkbox("Show Only Queue Items (Score ≥ 30)", value=True)

    display_df = filtered_df.copy()
    if only_in_queue:
        display_df = display_df[display_df["in_review_queue"] == 1]
    if search_query.strip():
        display_df = display_df[
            display_df["transaction_id"].str.contains(search_query.strip(), case=False)
            | display_df["sender_id"].str.contains(search_query.strip(), case=False)
        ]

    # Add Action Status column from session state
    display_df["analyst_status"] = display_df["transaction_id"].map(
        lambda tid: analyst_actions.get(tid, "Pending Review")
    )

    columns_to_render = [
        "transaction_id",
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount",
        "merchant_category",
        "upi_channel",
        "risk_score",
        "risk_level",
        "review_reason",
        "analyst_status",
    ]

    st.dataframe(
        display_df.sort_values("risk_score", ascending=False)[columns_to_render].head(100),
        use_container_width=True,
        column_config={
            "amount": st.column_config.NumberColumn("Amount (INR)", format="₹ %.2f"),
            "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%d pts"),
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="D MMM YYYY, HH:mm"),
            "analyst_status": st.column_config.TextColumn("Status"),
        },
        height=380,
    )

    # Interactive Action Panel
    st.markdown("#### ⚡ Quick Analyst Decision Terminal")
    act_col1, act_col2, act_col3 = st.columns([2, 2, 3])
    with act_col1:
        eligible_txns = display_df["transaction_id"].head(50).tolist()
        if eligible_txns:
            selected_tx = st.selectbox("Select Transaction to Triage", options=eligible_txns)
        else:
            selected_tx = None
            st.write("No matching transactions.")

    if selected_tx:
        tx_row = display_df[display_df["transaction_id"] == selected_tx].iloc[0]
        with act_col2:
            st.write(f"**Sender:** `{tx_row['sender_id']}` ➔ **Receiver:** `{tx_row['receiver_id']}`")
            st.write(f"**Amount:** INR {tx_row['amount']:,.2f} via **{tx_row['upi_channel']}**")
            st.write(f"**Reasons:** {tx_row['review_reason']}")
        with act_col3:
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("✅ Approve", key="btn_app"):
                    save_analyst_action(selected_tx, "Approved ✅")
                    st.success(f"{selected_tx} marked Approved!")
                    st.rerun()
            with b2:
                if st.button("⚠️ Verify KYC", key="btn_ver"):
                    save_analyst_action(selected_tx, "KYC Triggered ⚠️")
                    st.warning(f"Verification OTP sent for {selected_tx}")
                    st.rerun()
            with b3:
                if st.button("🚫 Block Account", key="btn_blk"):
                    save_analyst_action(selected_tx, "Blocked 🚫")
                    st.error(f"{selected_tx} sender frozen!")
                    st.rerun()

        # Per-transaction explainability: exactly why this score, rule by rule
        st.markdown("---")
        with st.expander(f"🧬 Explainability Breakdown — {selected_tx}", expanded=True):
            render_explanation(tx_row, dynamic_iqr_limit)


# ================= TAB 2: RISK ANALYTICS & PATTERNS =================
with tab_analytics:
    st.subheader("Payment Traffic Patterns & Risk Hotspots")

    an_row1_c1, an_row1_c2 = st.columns(2)

    with an_row1_c1:
        st.markdown("##### 🕒 Diurnal Hourly Activity vs Risk Spike Rate")
        hourly = (
            filtered_df.groupby("hour_of_day")
            .agg(
                total_txns=("transaction_id", "count"),
                flagged_txns=("in_review_queue", "sum"),
            )
            .reset_index()
        )
        hourly["risk_rate_pct"] = (hourly["flagged_txns"] / hourly["total_txns"] * 100).round(1)

        fig_hourly = go.Figure()
        fig_hourly.add_trace(
            go.Bar(
                x=hourly["hour_of_day"],
                y=hourly["total_txns"],
                name="Total Transactions",
                marker_color="#3b82f6",
                opacity=0.7,
            )
        )
        fig_hourly.add_trace(
            go.Scatter(
                x=hourly["hour_of_day"],
                y=hourly["risk_rate_pct"],
                name="Risk Alert Rate (%)",
                yaxis="y2",
                line=dict(color="#ef4444", width=3),
                mode="lines+markers",
            )
        )
        fig_hourly.update_layout(
            xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode="linear", dtick=2),
            yaxis=dict(title="Transaction Count"),
            yaxis2=dict(title="Alert Rate (%)", overlaying="y", side="right", showgrid=False),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with an_row1_c2:
        st.markdown("##### 💳 Channel Vulnerability Breakdown")
        channel_risk = (
            filtered_df.groupby("upi_channel")
            .agg(
                count=("transaction_id", "count"),
                high_risk=("in_review_queue", "sum"),
                total_amount=("amount", "sum"),
            )
            .reset_index()
        )
        fig_channel = px.pie(
            channel_risk,
            names="upi_channel",
            values="count",
            hole=0.45,
            color="upi_channel",
            color_discrete_map={"App": "#10b981", "QR": "#6366f1", "Collect Request": "#f59e0b"},
        )
        fig_channel.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=320)
        st.plotly_chart(fig_channel, use_container_width=True)

    an_row2_c1, an_row2_c2 = st.columns(2)

    with an_row2_c1:
        st.markdown("##### 📦 Transaction Amount Distribution by Category")
        fig_box = px.box(
            filtered_df,
            x="merchant_category",
            y="amount",
            color="merchant_category",
            log_y=True,
            title="Log Scale Amount Spread per Merchant Category",
        )
        fig_box.add_hline(
            y=dynamic_iqr_limit,
            line_dash="dash",
            line_color="red",
            annotation_text=f"IQR Cutoff: INR {dynamic_iqr_limit:,.0f}",
        )
        fig_box.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=330)
        st.plotly_chart(fig_box, use_container_width=True)

    with an_row2_c2:
        st.markdown("##### 📍 Geographic Risk Distribution")
        city_risk = (
            filtered_df.groupby("location")
            .agg(total=("transaction_id", "count"), queue=("in_review_queue", "sum"))
            .reset_index()
        )
        city_risk["queue_pct"] = (city_risk["queue"] / city_risk["total"] * 100).round(2)
        fig_city = px.bar(
            city_risk.sort_values("queue_pct", ascending=True),
            x="queue_pct",
            y="location",
            orientation="h",
            labels={"queue_pct": "Review Rate (%)", "location": "City"},
            color="queue_pct",
            color_continuous_scale="Reds",
        )
        fig_city.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=330)
        st.plotly_chart(fig_city, use_container_width=True)


# ================= TAB 3: ML VS RULE BENCHMARK =================
with tab_ml_benchmark:
    st.subheader("Model Evaluation: Explainable Rules vs. Unsupervised Isolation Forest")
    st.markdown(
        """
        Comparing our **Multi-Rule Composite Risk Engine** with an **Isolation Forest** anomaly detector trained on amount,
        temporal, and behavioral interaction features. Ground truth evaluates detection against known synthetic attack injections.
        """
    )

    # Compute metrics on filtered set
    y_true = filtered_df["risk_flag"]
    y_rule = filtered_df["in_review_queue"]
    y_ml = filtered_df["ml_prediction"]

    def calc_stats(yt, yp):
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return tp, fp, fn, prec, rec, f1

    r_tp, r_fp, r_fn, r_prec, r_rec, r_f1 = calc_stats(y_true, y_rule)
    m_tp, m_fp, m_fn, m_prec, m_rec, m_f1 = calc_stats(y_true, y_ml)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("#### 📐 Rule-Based Composite Engine")
        st.markdown(
            f"""
            - **Detection Recall:** **{r_rec:.1%}** ({r_tp:,} of {r_tp + r_fn:,} injected attacks caught)
            - **Precision:** **{r_prec:.1%}** ({r_fp:,} false alarms)
            - **F1 Score:** **{r_f1:.2f}**
            - **Explainability:** 100% Deterministic (exact rule rationale provided for every alert)
            - **Regulatory Suitability:** Complies with strict RBI / banking audit requirements
            """
        )

    with b_col2:
        st.markdown("#### 🌲 Isolation Forest (Unsupervised ML)")
        st.markdown(
            f"""
            - **Detection Recall:** **{m_rec:.1%}** ({m_tp:,} of {m_tp + m_fn:,} injected attacks caught)
            - **Precision:** **{m_prec:.1%}** ({m_fp:,} false alarms)
            - **F1 Score:** **{m_f1:.2f}**
            - **Explainability:** Black-box tree partitioning score
            - **Regulatory Suitability:** Requires SHAP/LIME post-hoc explainers before production deployment
            """
        )

    st.markdown("---")
    st.markdown("##### 🎯 Detection Agreement Scatter Plot (Amount vs. ML Anomaly Score)")
    sample_size = min(len(filtered_df), 1500)
    plot_sample = filtered_df.sample(sample_size, random_state=42) if len(filtered_df) > sample_size else filtered_df

    fig_scatter = px.scatter(
        plot_sample,
        x="amount",
        y="ml_anomaly_score",
        color="risk_level",
        color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#10b981"},
        hover_data=["transaction_id", "sender_id", "upi_channel", "review_reason"],
        log_x=True,
        labels={"amount": "Amount INR (Log Scale)", "ml_anomaly_score": "ML Anomaly Score (0-100)"},
        title=f"Sample of {len(plot_sample):,} Transactions: Color Coded by Rule Risk Tier",
    )
    fig_scatter.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)


# ================= TAB 4: SQL LAB =================
with tab_sql:
    st.subheader("SQL Analytics Lab (SQLite Engine)")
    st.caption("Directly test and run SQL queries against the active dataset.")

    conn = get_sqlite_connection(raw_df)

    queries = {
        "1. Merchant Category Volume & Total Spend": """
SELECT 
    merchant_category, 
    COUNT(*) AS total_transactions, 
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_ticket_size
FROM upi_transactions
GROUP BY merchant_category
ORDER BY total_amount DESC;
        """,
        "2. Hourly UPI Traffic & Diurnal Cycle Breakdown": """
SELECT 
    strftime('%H', timestamp) AS hour_of_day, 
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS avg_amount,
    SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) AS risk_flagged_count
FROM upi_transactions
GROUP BY hour_of_day
ORDER BY hour_of_day ASC;
        """,
        "3. High-Risk Payments by City": """
SELECT 
    location, 
    COUNT(*) AS total_txns,
    SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) AS high_risk_txns,
    ROUND(100.0 * SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS risk_percentage
FROM upi_transactions
GROUP BY location
ORDER BY high_risk_txns DESC;
        """,
        "4. Channel Vulnerability Breakdown": """
SELECT 
    upi_channel,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS total_volume,
    SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) AS flagged_count,
    ROUND(100.0 * SUM(CASE WHEN risk_flag = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS incident_rate_pct
FROM upi_transactions
GROUP BY upi_channel
ORDER BY incident_rate_pct DESC;
        """,
        "5. Sender Accounts with Rapid Consecutive Transactions": """
WITH ordered_transactions AS (
    SELECT sender_id, timestamp, amount,
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
        """,
    }

    selected_query_name = st.selectbox("Select Pre-configured SQL Query", options=list(queries.keys()))
    default_sql = queries[selected_query_name].strip()

    sql_text = st.text_area("SQL Editor", value=default_sql, height=140)

    if st.button("▶️ Execute Query", key="btn_run_sql"):
        try:
            sql_result = run_read_only_query(conn, sql_text)
            st.success(f"Query returned {len(sql_result):,} rows.")
            st.dataframe(sql_result, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Execution Error: {e}")


# ================= TAB 5: SCORE YOUR OWN CSV =================
with tab_upload:
    st.subheader("Score Your Own Transaction CSV")
    st.caption(
        "**Required columns:** `timestamp`, `amount`, `sender_id`, `receiver_id`, `upi_channel` — "
        "**optional:** `merchant_category` + `device_type` (enables the Isolation Forest benchmark). "
        f"No `transaction_id`? One is generated for you. Sample file to try: `{RAW_DATA_FILE.name}`."
    )

    uploaded_file = st.file_uploader("Upload a transactions CSV", type=["csv"], key="csv_uploader")

    if uploaded_file is None:
        st.info(
            "Upload a CSV to have the explainable rule engine score it live "
            "with your current IQR multiplier setting from the sidebar."
        )
    else:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not parse the CSV: {exc}")
            uploaded_df = None

        if uploaded_df is not None:
            upload_problems = validate_transaction_upload(uploaded_df)
            if upload_problems:
                for problem in upload_problems:
                    st.error(problem)
            else:
                try:
                    scored_df, upload_iqr_limit = score_upload_cached(uploaded_df, float(iqr_mult))
                except Exception as exc:
                    st.error(f"Scoring failed: {exc}")
                    scored_df = None

                if scored_df is not None:
                    # KPI row for the uploaded batch
                    k1, k2, k3, k4 = st.columns(4)
                    queue_rate = (scored_df["in_review_queue"] == 1).mean() * 100
                    k1.metric("Rows Scored", f"{len(scored_df):,}")
                    k2.metric(
                        "In Review Queue",
                        f"{int((scored_df['in_review_queue'] == 1).sum()):,}",
                        f"{queue_rate:.1f}% queue rate",
                    )
                    k3.metric("High-Risk Alerts", f"{int((scored_df['risk_level'] == 'High').sum()):,}", delta_color="inverse")
                    k4.metric("IQR Cutoff", f"INR {upload_iqr_limit:,.2f}")

                    if "ml_prediction" not in scored_df.columns:
                        st.info(
                            "Isolation Forest benchmark skipped — add `merchant_category` and "
                            "`device_type` columns to enable it."
                        )

                    upload_cols = [
                        col
                        for col in (
                            "transaction_id", "timestamp", "sender_id", "receiver_id", "amount",
                            "upi_channel", "risk_score", "risk_level", "review_reason",
                        )
                        if col in scored_df.columns
                    ]
                    st.dataframe(
                        scored_df.sort_values("risk_score", ascending=False)[upload_cols].head(200),
                        use_container_width=True,
                        column_config={
                            "amount": st.column_config.NumberColumn("Amount (INR)", format="₹ %.2f"),
                            "risk_score": st.column_config.ProgressColumn(
                                "Risk Score", min_value=0, max_value=100, format="%d pts"
                            ),
                            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="D MMM YYYY, HH:mm"),
                        },
                        height=380,
                    )

                    # Explainability for any row in the uploaded batch
                    st.markdown("#### 🧬 Explain a Scored Transaction")
                    score_options = (
                        scored_df.sort_values("risk_score", ascending=False)["transaction_id"].head(50).tolist()
                    )
                    chosen_upload_tx = st.selectbox(
                        "Transaction to explain", options=score_options, key="upload_explain"
                    )
                    if chosen_upload_tx:
                        upload_row = scored_df[scored_df["transaction_id"] == chosen_upload_tx].iloc[0]
                        render_explanation(upload_row, upload_iqr_limit)

                    st.download_button(
                        "⬇️ Download Scored CSV",
                        data=scored_df.to_csv(index=False).encode("utf-8"),
                        file_name="scored_transactions.csv",
                        mime="text/csv",
                    )
