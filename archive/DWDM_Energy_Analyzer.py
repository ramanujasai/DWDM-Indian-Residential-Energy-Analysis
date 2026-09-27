import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# INDIAN RESIDENTIAL ENERGY CONSUMPTION PATTERN ANALYSIS
# DWDM LAB PROJECT - 23CS3551
# ============================================================

st.set_page_config(
    page_title="Indian Residential Energy Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
RESULTS = BASE / "results"

FEATURES = [
    "mean_consumption",
    "max_consumption",
    "std_consumption",
    "median_consumption",
    "peak_to_average",
    "daytime_avg",
    "evening_avg",
    "night_avg",
    "weekday_avg",
    "weekend_avg",
    "weekday_weekend_difference",
]

# Project results obtained from the analyzed CEEW Bareilly 2020 dataset.
CLUSTER_SIZES = {0: 11, 1: 6, 2: 15}

CLUSTER_PROFILES = {
    0: {"mean": 0.006317, "max": 0.062000, "std": 0.0065, "median": 0.004273,
        "peak_avg": 11.9616, "day": 0.0059, "evening": 0.0071, "night": 0.0065},
    1: {"mean": 0.035670, "max": 0.231833, "std": 0.0356, "median": 0.022500,
        "peak_avg": 6.6715, "day": 0.0346, "evening": 0.0382, "night": 0.0360},
    2: {"mean": 0.020892, "max": 0.188133, "std": 0.0224, "median": 0.014467,
        "peak_avg": 8.9705, "day": 0.0191, "evening": 0.0235, "night": 0.0223},
}

CLUSTER_NAMES = {
    0: "Lower Consumption Pattern",
    1: "Higher Consumption Pattern",
    2: "Moderate Consumption Pattern",
}

# ============================================================
# DARK THEME
# ============================================================

st.markdown("""
<style>
:root {
    --bg: #080d16;
    --panel: #0f1724;
    --panel2: #121d2d;
    --border: #223047;
    --text: #e8eef7;
    --muted: #8fa0b7;
    --blue: #38bdf8;
    --blue2: #2563eb;
    --cyan: #22d3ee;
}

.stApp {
    background: #080d16;
    color: #e8eef7;
}

.main .block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #060a12;
    border-right: 1px solid #1b2638;
}

section[data-testid="stSidebar"] * {
    color: #dce6f3 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #f3f7fc !important;
}

p, li, label, .stMarkdown, .stCaption {
    color: #b8c5d6;
}

.hero {
    background: linear-gradient(135deg, #0e1726 0%, #102a43 55%, #0b1c30 100%);
    border: 1px solid #243955;
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 12px 35px rgba(0,0,0,.28);
}

.hero-kicker {
    color: #38bdf8;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .16em;
    margin-bottom: .5rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.25rem;
    color: #f8fbff !important;
}

.hero p {
    margin: .55rem 0 0;
    color: #aebed1 !important;
    font-size: 1rem;
}

.section-label {
    color: #38bdf8;
    font-size: .75rem;
    font-weight: 800;
    letter-spacing: .12em;
    margin-bottom: .45rem;
}

.panel {
    background: #0f1724;
    border: 1px solid #223047;
    border-radius: 15px;
    padding: 1.15rem;
    margin-bottom: 1rem;
}

.panel h3 {
    margin-top: 0;
}

.kpi {
    background: #0f1724;
    border: 1px solid #223047;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    min-height: 105px;
}

.kpi-label {
    color: #8092aa;
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .07em;
}

.kpi-value {
    color: #f1f7ff;
    font-size: 1.65rem;
    font-weight: 750;
    margin-top: .35rem;
}

.kpi-sub {
    color: #7588a1;
    font-size: .78rem;
    margin-top: .2rem;
}

.cluster-card {
    background: #101a29;
    border: 1px solid #263952;
    border-radius: 14px;
    padding: 1.1rem;
}

.cluster-number {
    color: #38bdf8;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .12em;
}

.cluster-card h3 {
    margin: .35rem 0 .7rem;
}

.metric-box {
    background: #0b1320;
    border: 1px solid #1f2e44;
    border-radius: 11px;
    padding: .75rem;
}

.metric-box .label {
    color: #7e91a9;
    font-size: .72rem;
}

.metric-box .value {
    color: #e9f2fc;
    font-weight: 700;
    margin-top: .2rem;
}

.result {
    background: linear-gradient(135deg, #101d2e, #0d1826);
    border: 1px solid #2a4564;
    border-radius: 16px;
    padding: 1.4rem;
    margin: 1rem 0;
}

.result-kicker {
    color: #38bdf8;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .12em;
}

.result h2 {
    margin: .35rem 0 .5rem;
}

.result p {
    color: #aebed1;
}

div[data-testid="stMetric"] {
    background: #0f1724 !important;
    border: 1px solid #223047 !important;
    border-radius: 13px !important;
    padding: .8rem !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    color: #8fa0b7 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] > div {
    color: #eef6ff !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] {
    background: #0f1724 !important;
    border-color: #2a3a51 !important;
}

input, textarea {
    color: #eef6ff !important;
    -webkit-text-fill-color: #eef6ff !important;
}

div[data-baseweb="select"] span {
    color: #eef6ff !important;
}

section[data-testid="stFileUploaderDropzone"] {
    background: #0f1724 !important;
    border: 1px dashed #38506d !important;
}

section[data-testid="stFileUploaderDropzone"] * {
    color: #c8d5e4 !important;
}

.stButton > button {
    background: #172438 !important;
    color: #edf6ff !important;
    border: 1px solid #2d4563 !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
}

.stButton > button:hover {
    border-color: #38bdf8 !important;
    color: #ffffff !important;
}

.stButton > button[kind="primary"] {
    background: #1769aa !important;
    border-color: #2196d2 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #223047;
    border-radius: 10px;
}

[data-testid="stTabs"] button {
    color: #91a4bb !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
}

.stAlert {
    background: #0f1724 !important;
    border-color: #263b55 !important;
}

hr {
    border-color: #1e2b3e !important;
}

code {
    color: #67e8f9 !important;
    background: #0b1320 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_final_model():
    scaler_path = MODELS / "final_kmeans_scaler.pkl"
    kmeans_path = MODELS / "final_kmeans.pkl"

    if not scaler_path.exists() or not kmeans_path.exists():
        return None, None

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    return scaler, kmeans


scaler, kmeans = load_final_model()


# ============================================================
# DATA PROCESSING
# ============================================================

def required_columns(df):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    meter = lookup.get("meter")
    timestamp = lookup.get("x_timestamp")
    consumption = lookup.get("t_kwh")

    if meter is None or timestamp is None or consumption is None:
        raise ValueError(
            "The CSV must contain these columns: "
            "meter, x_Timestamp, t_kWh"
        )

    return meter, timestamp, consumption


def build_features(raw):
    meter_col, timestamp_col, consumption_col = required_columns(raw)

    df = raw[[meter_col, timestamp_col, consumption_col]].copy()
    df.columns = ["meter", "timestamp", "consumption"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")

    df = df.dropna(subset=["meter", "timestamp", "consumption"])
    df = df[df["consumption"] >= 0].copy()

    if df.empty:
        raise ValueError("No valid non-negative consumption readings were found.")

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek

    rows = []

    for meter, g in df.groupby("meter", sort=True):
        s = g["consumption"]

        mean_v = s.mean()
        max_v = s.max()

        daytime = g.loc[
            (g["hour"] >= 6) & (g["hour"] < 18), "consumption"
        ]
        evening = g.loc[
            (g["hour"] >= 18) & (g["hour"] < 22), "consumption"
        ]
        night = g.loc[
            (g["hour"] >= 22) | (g["hour"] < 6), "consumption"
        ]
        weekday = g.loc[g["weekday"] < 5, "consumption"]
        weekend = g.loc[g["weekday"] >= 5, "consumption"]

        weekday_avg = weekday.mean()
        weekend_avg = weekend.mean()

        rows.append({
            "meter": meter,
            "mean_consumption": mean_v,
            "max_consumption": max_v,
            "std_consumption": s.std(),  # ddof=1, same as training
            "median_consumption": s.median(),
            "peak_to_average": max_v / mean_v if mean_v else 0,
            "daytime_avg": daytime.mean(),
            "evening_avg": evening.mean(),
            "night_avg": night.mean(),
            "weekday_avg": weekday_avg,
            "weekend_avg": weekend_avg,
            "weekday_weekend_difference": abs(
                weekday_avg - weekend_avg
            ),
        })

    result = pd.DataFrame(rows)
    result[FEATURES] = (
        result[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return result


def build_time_profile(raw):
    meter_col, timestamp_col, consumption_col = required_columns(raw)

    df = raw[[meter_col, timestamp_col, consumption_col]].copy()
    df.columns = ["meter", "timestamp", "consumption"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df = df.dropna()
    df = df[df["consumption"] >= 0].copy()
    df["hour"] = df["timestamp"].dt.hour

    rows = []

    for meter, g in df.groupby("meter", sort=True):
        total = g["consumption"].sum()

        def share(mask):
            amount = g.loc[mask, "consumption"].sum()
            return amount / total if total > 0 else 0

        morning = share((g["hour"] >= 6) & (g["hour"] < 12))
        afternoon = share((g["hour"] >= 12) & (g["hour"] < 18))
        evening = share((g["hour"] >= 18) & (g["hour"] < 22))
        night = share((g["hour"] >= 22) | (g["hour"] < 6))

        shares = {
            "Morning": morning,
            "Afternoon": afternoon,
            "Evening": evening,
            "Night": night,
        }

        rows.append({
            "meter": meter,
            **{f"{k.lower()}_share": v for k, v in shares.items()},
            "dominant_period": max(shares, key=shares.get),
        })

    return pd.DataFrame(rows)


def analyze_csv(raw):
    features = build_features(raw)

    if scaler is None or kmeans is None:
        raise FileNotFoundError(
            "Final K-Means model files are missing from the models folder."
        )

    labels = kmeans.predict(
        scaler.transform(features[FEATURES])
    )

    result = features.copy()
    result["cluster"] = labels.astype(int)

    time_profile = build_time_profile(raw)

    return result.merge(
        time_profile,
        on="meter",
        how="left",
    )



# ============================================================
# DATA-BASED CONSUMPTION REDUCTION ANALYSIS
# ============================================================

def build_reduction_analysis(row):
    periods = {
        "Morning": float(row.get("morning_share", 0)),
        "Afternoon": float(row.get("afternoon_share", 0)),
        "Evening": float(row.get("evening_share", 0)),
        "Night": float(row.get("night_share", 0)),
    }

    dominant_period = max(periods, key=periods.get)
    dominant_share = periods[dominant_period]
    cluster = int(row["cluster"])
    cp = CLUSTER_PROFILES[cluster]

    comparisons = {
        "Mean consumption": (float(row["mean_consumption"]), cp["mean"]),
        "Peak consumption": (float(row["max_consumption"]), cp["max"]),
        "Variability": (float(row["std_consumption"]), cp["std"]),
        "Peak / average": (float(row["peak_to_average"]), cp["peak_avg"]),
        "Daytime average": (float(row["daytime_avg"]), cp["day"]),
        "Evening average": (float(row["evening_avg"]), cp["evening"]),
        "Night average": (float(row["night_avg"]), cp["night"]),
    }

    suggestions = []

    if dominant_share >= 0.35:
        suggestions.append({
            "priority": "High",
            "area": f"{dominant_period} consumption",
            "finding": (
                f"{dominant_period} accounts for {dominant_share * 100:.1f}% "
                "of the recorded consumption."
            ),
            "action": (
                f"Review the loads operating during {dominant_period.lower()} "
                "and switch off unnecessary loads during this period."
            ),
        })
    elif dominant_share >= 0.30:
        suggestions.append({
            "priority": "Medium",
            "area": f"{dominant_period} consumption",
            "finding": (
                f"{dominant_period} is the largest usage period at "
                f"{dominant_share * 100:.1f}% of recorded consumption."
            ),
            "action": (
                f"Check whether non-essential loads can be reduced or shifted "
                f"outside the {dominant_period.lower()} period."
            ),
        })

    user_mean = float(row["mean_consumption"])
    if cp["mean"] > 0 and user_mean > cp["mean"] * 1.15:
        pct = (user_mean / cp["mean"] - 1) * 100
        suggestions.append({
            "priority": "High",
            "area": "Typical consumption level",
            "finding": (
                f"Mean consumption is {pct:.1f}% above the average "
                "of the assigned cluster."
            ),
            "action": (
                "Focus first on the periods with the largest consumption "
                "share and identify avoidable or unnecessary loads."
            ),
        })

    user_peak = float(row["max_consumption"])
    if cp["max"] > 0 and user_peak > cp["max"] * 1.20:
        pct = (user_peak / cp["max"] - 1) * 100
        suggestions.append({
            "priority": "High",
            "area": "Peak intervals",
            "finding": (
                f"The maximum recorded interval is {pct:.1f}% above "
                "the typical peak of the assigned cluster."
            ),
            "action": (
                "Inspect the timestamps around the highest-consumption "
                "intervals and look for simultaneous or unnecessary loads."
            ),
        })

    user_ratio = float(row["peak_to_average"])
    if cp["peak_avg"] > 0 and user_ratio > cp["peak_avg"] * 1.20:
        suggestions.append({
            "priority": "Medium",
            "area": "Consumption spikes",
            "finding": (
                f"The peak-to-average ratio ({user_ratio:.2f}) is noticeably "
                f"higher than the cluster profile ({cp['peak_avg']:.2f})."
            ),
            "action": (
                "Investigate short high-consumption periods rather than "
                "only looking at the overall average."
            ),
        })

    time_checks = [
        ("Afternoon", "daytime_avg", cp["day"]),
        ("Evening", "evening_avg", cp["evening"]),
        ("Night", "night_avg", cp["night"]),
    ]

    for period, key, reference in time_checks:
        actual = float(row[key])
        if reference > 0 and actual > reference * 1.20:
            pct = (actual / reference - 1) * 100
            suggestions.append({
                "priority": "Medium",
                "area": f"{period} usage",
                "finding": (
                    f"Average {period.lower()} consumption is "
                    f"{pct:.1f}% above the assigned cluster profile."
                ),
                "action": (
                    f"Review which loads are active during the "
                    f"{period.lower()} period and reduce avoidable usage."
                ),
            })

    if not suggestions:
        suggestions.append({
            "priority": "Normal",
            "area": "Overall profile",
            "finding": (
                "No strong reduction signal crossed the project's "
                "comparison thresholds."
            ),
            "action": (
                "Continue monitoring the dominant period and the highest "
                "consumption intervals for future opportunities."
            ),
        })

    return dominant_period, dominant_share, comparisons, suggestions


def render_reduction_analysis(row):
    dominant, share, comparisons, suggestions = build_reduction_analysis(row)

    st.markdown("### Where can this consumption be reduced?")

    st.markdown(
        f"""
        <div class="result">
            <div class="result-kicker">DATA-BASED FINDING</div>
            <h2>{dominant} is the dominant usage period</h2>
            <p>
                <b>{share * 100:.1f}%</b> of this meter's recorded consumption
                occurs during the {dominant.lower()} period.
                Recommendations below are generated from the uploaded
                smart-meter readings and the assigned K-Means cluster.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    order = {"High": 0, "Medium": 1, "Normal": 2}
    for item in sorted(suggestions, key=lambda x: order[x["priority"]]):
        st.markdown(
            f"""
            <div class="panel">
                <div style="color:#38bdf8;font-size:.72rem;
                            font-weight:800;letter-spacing:.10em;">
                    {item["priority"].upper()} PRIORITY • {item["area"].upper()}
                </div>
                <h3 style="margin:.4rem 0;">{item["finding"]}</h3>
                <p style="margin-bottom:0;">{item["action"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Your profile vs. similar households")

    comparison_df = pd.DataFrame([
        {
            "Metric": metric,
            "Your value": values[0],
            "Cluster profile": values[1],
            "Difference": values[0] - values[1],
        }
        for metric, values in comparisons.items()
    ])

    st.dataframe(
        comparison_df.style.format({
            "Your value": "{:.5f}",
            "Cluster profile": "{:.5f}",
            "Difference": "{:+.5f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The smart-meter data identifies consumption patterns and time periods, "
        "but does not identify individual appliances. Therefore the application "
        "does not claim that a particular appliance caused a peak."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:.5rem 0 1rem;">
        <div style="font-size:1.25rem;font-weight:800;color:#f5f9ff;">
            ⚡ ENERGY ANALYZER
        </div>
        <div style="font-size:.72rem;color:#71849d;margin-top:.25rem;">
            DWDM LAB • 23CS3551
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "PROJECT MODULES",
        [
            "Dashboard",
            "User Dataset",
            "Technical Analysis",
            "Consumer Analysis",
            "About Project",
        ],
        label_visibility="visible",
    )

    st.markdown("---")
    st.caption("DWDM • Smart-meter analysis")
    st.caption("Current-user profiling")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">DWDM LAB • 23CS3551</div>
        <h1>Residential Energy Consumption Analyzer</h1>
        <p>
            Analyze the current user's smart-meter readings and identify
            consumption patterns using the trained Data Mining model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "consumer_result" in st.session_state:
        result = st.session_state["consumer_result"]

        if len(result) == 1:
            row = result.iloc[0]
            cluster = int(row["cluster"])

            a, b, c, d = st.columns(4)
            a.metric("User / Meter", str(row["meter"]))
            b.metric("Predicted cluster", cluster)
            c.metric("Mean consumption", f"{row['mean_consumption']:.5f}")
            d.metric("Dominant period", str(row["dominant_period"]))

            st.markdown(
                f"""
                <div class="result">
                    <div class="result-kicker">CURRENT USER PROFILE</div>
                    <h2>{CLUSTER_NAMES[cluster]}</h2>
                    <p>
                        Meter: <b>{row['meter']}</b>
                        &nbsp; • &nbsp; Cluster: <b>{cluster}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_reduction_analysis(row)
        else:
            st.metric("Meters analyzed", len(result))
            st.dataframe(
                result[["meter", "mean_consumption", "max_consumption", "cluster"]]
                .rename(columns={
                    "meter": "Meter",
                    "mean_consumption": "Mean Consumption",
                    "max_consumption": "Peak Consumption",
                    "cluster": "Cluster",
                }),
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.markdown("""
        <div class="panel">
            <h3>No current user loaded</h3>
            <p>
                Open <b>Consumer Analysis</b> and provide the current user's
                smart-meter CSV. The trained scaler and K-Means model are
                already included in the project, so the original training
                CSV is not required.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Current-user analysis workflow")

    cols = st.columns(5)
    steps = [
        ("01", "User CSV", "Current user's readings"),
        ("02", "Features", "11 project features"),
        ("03", "Scaling", "Saved StandardScaler"),
        ("04", "Prediction", "Saved K-Means"),
        ("05", "Advice", "Consumption reduction analysis"),
    ]

    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="panel">
                    <div style="color:#38bdf8;font-weight:800;">{num}</div>
                    <h4>{title}</h4>
                    <p style="font-size:.82rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DATASET
# ============================================================

# ============================================================
# PREPROCESSING
# ============================================================

elif page == "Technical Analysis":

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">DWDM MODEL • 23CS3551</div>
        <h1>Technical Analysis</h1>
        <p>A compact view of the model used by the current-user analyzer.</p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.metric("Features", "11")
    b.metric("K-Means clusters", "3")
    c.metric("Silhouette", "0.4197")

    st.markdown("### Model pipeline")
    st.markdown("""
    <div class="panel">
        <b>Smart-meter CSV</b> → <b>Feature Engineering</b> →
        <b>StandardScaler</b> → <b>K-Means (K=3)</b> →
        <b>Consumption Pattern</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Features used by K-Means")
    feature_df = pd.DataFrame({
        "Feature": FEATURES,
        "Role": [
            "Typical consumption",
            "Maximum consumption",
            "Variability",
            "Typical midpoint",
            "Peak relative to average",
            "Daytime usage",
            "Evening usage",
            "Night usage",
            "Weekday usage",
            "Weekend usage",
            "Weekday/weekend difference",
        ],
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.markdown("### Cluster evaluation")
    eval_df = pd.DataFrame({
        "K": [2, 3, 4, 5, 6],
        "Silhouette": [0.4114, 0.4197, 0.4133, 0.3092, 0.3034],
        "Davies-Bouldin": [0.8363, 0.8458, 0.6762, 0.7904, 0.8763],
    })
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    p1, p2 = st.columns(2)
    with p1:
        path = RESULTS / "silhouette_plot.png"
        if path.exists():
            st.image(str(path), caption="Silhouette Analysis", use_container_width=True)
    with p2:
        path = RESULTS / "pca_clusters.png"
        if path.exists():
            st.image(str(path), caption="PCA Cluster Visualization", use_container_width=True)

    st.caption(
        "Classification models and clustering comparisons are retained in the "
        "project files for academic evaluation; the user-facing workflow uses "
        "the saved K-Means model."
    )
# ============================================================
# CLUSTERING
# ============================================================

# ============================================================
# CLASSIFICATION
# ============================================================

# ============================================================
# CONSUMER ANALYSIS
# ============================================================

elif page == "Consumer Analysis":

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">MODULE 05 • MODEL APPLICATION</div>
        <h1>Smart-Meter Consumer Analysis</h1>
        <p>
            Upload smart-meter readings. The same feature engineering,
            scaling and K-Means model used in the project will be applied.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
    <b>Required CSV columns</b><br>
    <code>meter</code> &nbsp; <code>x_Timestamp</code> &nbsp; <code>t_kWh</code>
    <br><br>
    The prediction is based on the uploaded smart-meter readings.
    No manually entered consumption values are used to assign a cluster.
    </div>
    """, unsafe_allow_html=True)

    default_user = BASE / "user.csv"

    if default_user.exists():
        st.success(
            "user.csv detected in the project folder. "
            "This file can be analyzed directly."
        )

        if st.button(
            "Analyze user.csv",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Processing the current user's smart-meter readings..."):
                    raw = pd.read_csv(default_user)
                    result = analyze_csv(raw)

                st.session_state["consumer_result"] = result
                st.success(
                    f"Analysis complete. {len(result)} meter profile(s) were generated."
                )

            except Exception as exc:
                st.error(str(exc))

        st.caption("Upload another CSV below if you want to analyze a different user.")

    uploaded = st.file_uploader(
        "Upload another user's smart-meter CSV",
        type=["csv"],
        key="consumer_upload",
    )

    if uploaded:

        if st.button(
            "Analyze uploaded user",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Processing the current user's smart-meter readings..."):
                    raw = pd.read_csv(uploaded)
                    result = analyze_csv(raw)

                st.session_state["consumer_result"] = result
                st.success(
                    f"Analysis complete. {len(result)} meter profile(s) were generated."
                )

            except Exception as exc:
                st.error(str(exc))

    if "consumer_result" in st.session_state:

        result = st.session_state["consumer_result"]

    
        summary = result[
            ["meter", "mean_consumption", "max_consumption", "cluster"]
        ].copy()
        summary["Pattern"] = summary["cluster"].map(CLUSTER_NAMES)

        summary.columns = [
            "Meter",
            "Mean Consumption",
            "Peak Consumption",
            "Cluster",
            "Pattern",
        ]

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Current user")

        selected_meter = st.selectbox(
            "Meter",
            result["meter"].astype(str).tolist(),
            label_visibility="collapsed",
        )

        row = result[
            result["meter"].astype(str) == selected_meter
        ].iloc[0]

        cluster = int(row["cluster"])

        st.markdown(
            f'<div class="result">'
            f'<div class="result-kicker">K-MEANS PREDICTION</div>'
            f'<h2>{CLUSTER_NAMES[cluster]}</h2>'
            f'<p>Meter: <b>{selected_meter}</b> &nbsp; • &nbsp; '
            f'Cluster: <b>{cluster}</b></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        a, b, c, d = st.columns(4)

        a.metric("Mean", f"{row['mean_consumption']:.5f}")
        b.metric("Peak", f"{row['max_consumption']:.5f}")
        c.metric("Std. deviation", f"{row['std_consumption']:.5f}")
        d.metric("Peak / Average", f"{row['peak_to_average']:.2f}")

        st.markdown("### Time-of-day profile")

        time_df = pd.DataFrame({
            "Period": ["Morning", "Afternoon", "Evening", "Night"],
            "Share (%)": [
                row["morning_share"] * 100,
                row["afternoon_share"] * 100,
                row["evening_share"] * 100,
                row["night_share"] * 100,
            ],
        }).set_index("Period")

        st.bar_chart(time_df)

        dominant = row["dominant_period"]

        st.markdown(
            f'<div class="panel">'
            f'<div class="section-label">USAGE TIMING</div>'
            f'<h3>Highest consumption period: {dominant}</h3>'
            f'<p>The household profile contains the largest share of '
            f'consumption during this period.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        render_reduction_analysis(row)



# ============================================================
# ABOUT
# ============================================================

else:

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">PROJECT INFORMATION • 23CS3551</div>
        <h1>About the Project</h1>
        <p>Residential energy consumption pattern analysis using Data Mining.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Objective")
    st.write(
        "Identify groups of residential consumers with similar electricity "
        "consumption behaviour and analyze the current user's profile."
    )

    st.markdown("### Method")
    st.write(
        "The project engineers consumer-level features from smart-meter "
        "readings, standardizes them, and applies K-Means clustering."
    )

    st.markdown("### Current-user analysis")
    st.write(
        "A new user's CSV is converted into the same feature representation, "
        "passed through the saved scaler and K-Means model, and then analyzed "
        "for time-of-day usage and possible consumption-reduction areas."
    )

    st.markdown("### Important limitation")
    st.write(
        "The smart-meter data identifies consumption patterns and timing, "
        "not individual appliances. Recommendations therefore focus on "
        "measured usage periods and statistical patterns."
    )
    