import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import os

# ============================================================
# USER PROFILE K-MEANS MODEL
# ============================================================
# This script uses the consumer_features.csv produced by
# analysis_time_features.py.
#
# User-facing features:
#   1. avg_daily_kwh
#   2. morning_share       (06:00-12:00)
#   3. afternoon_share     (12:00-18:00)
#   4. evening_share       (18:00-22:00)
#   5. night_share         (22:00-06:00)
#
# The time shares are directly useful for the later
# recommendation system: the app can identify the period
# in which a household uses the largest share of electricity.
# ============================================================

INPUT = "consumer_features.csv"
OUTPUT_FEATURES = "user_profile_features.csv"
K_RESULTS = "k_selection_results.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "avg_daily_kwh",
    "morning_share",
    "afternoon_share",
    "evening_share",
    "night_share"
]

print("Loading consumer features...")
df = pd.read_csv(INPUT)

required = [
    "meter",
    "mean_consumption",
    "morning_share",
    "afternoon_share",
    "evening_share",
    "night_share"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 3-minute readings -> 20 readings/hour -> 480 readings/day.
# mean_consumption is kWh per 3-minute interval, so multiplying
# by 480 gives an estimated average daily kWh while remaining
# robust to different amounts of missing data between meters.
df["avg_daily_kwh"] = df["mean_consumption"] * 480

profile_df = df[["meter"] + FEATURES].copy()

# Convert shares to percentages for the saved human-readable file.
for col in [
    "morning_share",
    "afternoon_share",
    "evening_share",
    "night_share"
]:
    profile_df[col] = profile_df[col].clip(0, 1)

profile_df.to_csv(OUTPUT_FEATURES, index=False)

X = profile_df[FEATURES].values

# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# TEST DIFFERENT VALUES OF K
# ============================================================

print("\nTesting K-Means values...")

results = []

for k in range(2, 7):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    results.append({
        "k": k,
        "silhouette_score": silhouette,
        "davies_bouldin_score": db,
        "cluster_sizes": str(
            np.bincount(labels).tolist()
        )
    })

    print(
        f"K={k} | "
        f"Silhouette={silhouette:.4f} | "
        f"Davies-Bouldin={db:.4f} | "
        f"Sizes={np.bincount(labels).tolist()}"
    )

results_df = pd.DataFrame(results)
results_df.to_csv(K_RESULTS, index=False)

# ============================================================
# FINAL MODEL
# ============================================================
# We retain K=3 to keep the project's three-pattern design
# and compare the resulting profiles rather than selecting
# a K solely from one metric.

FINAL_K = 3

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=10
)

profile_df["cluster"] = kmeans.fit_predict(X_scaled)

# ============================================================
# CREATE HUMAN-READABLE PATTERN PROFILES
# ============================================================

profile_summary = (
    profile_df
    .groupby("cluster")[FEATURES]
    .mean()
    .reset_index()
)

profile_summary["households"] = (
    profile_df.groupby("cluster").size().values
)

# Determine the dominant time period for each cluster.
period_cols = [
    "morning_share",
    "afternoon_share",
    "evening_share",
    "night_share"
]

period_names = {
    "morning_share": "Morning",
    "afternoon_share": "Afternoon",
    "evening_share": "Evening",
    "night_share": "Night"
}

dominant_periods = []

for _, row in profile_summary.iterrows():
    dominant = max(
        period_cols,
        key=lambda col: row[col]
    )
    dominant_periods.append(period_names[dominant])

profile_summary["dominant_period"] = dominant_periods

# Human-readable pattern name based on daily consumption.
daily_values = profile_summary["avg_daily_kwh"]

low_cut = daily_values.quantile(0.33)
high_cut = daily_values.quantile(0.67)

pattern_names = []

for value in daily_values:
    if value <= low_cut:
        name = "Lower Daily Consumption"
    elif value >= high_cut:
        name = "Higher Daily Consumption"
    else:
        name = "Moderate Daily Consumption"

    pattern_names.append(name)

profile_summary["pattern_name"] = pattern_names

profile_summary.to_csv(
    "pattern_profiles.csv",
    index=False
)

# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "user_profile_scaler.pkl")
)

joblib.dump(
    kmeans,
    os.path.join(MODEL_DIR, "user_profile_kmeans.pkl")
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n==============================================")
print("USER PROFILE K-MEANS COMPLETE")
print("==============================================")

print("\nFeatures used:")
for feature in FEATURES:
    print("-", feature)

print("\nFinal K:", FINAL_K)

print("\nCluster sizes:")
print(profile_df["cluster"].value_counts().sort_index())

print("\nPattern profiles:")
print(
    profile_summary[
        [
            "cluster",
            "pattern_name",
            "households",
            "avg_daily_kwh",
            "morning_share",
            "afternoon_share",
            "evening_share",
            "night_share",
            "dominant_period"
        ]
    ].to_string(index=False)
)

print("\nFiles created:")
print("-", OUTPUT_FEATURES)
print("-", K_RESULTS)
print("-", "pattern_profiles.csv")
print("-", "models/user_profile_scaler.pkl")
print("-", "models/user_profile_kmeans.pkl")

print("\nNext step:")
print("Use dominant_period in the application to generate")
print("time-specific electricity-saving recommendations.")
