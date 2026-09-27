import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

# ============================================================
# MODEL EXPERIMENT V2
# ============================================================
# Goal:
# 1. Calculate average daily energy directly from timestamps.
# 2. Keep the four time-of-day consumption shares.
# 3. Compare feature sets instead of forcing the 5-feature
#    model to produce useful clusters.
# 4. Preserve the time-period features for the recommendation
#    layer.
#
# The experiment compares:
#
# A. Original 11 clustering features
# B. Original 11 + 4 time-share features
# C. User-facing 5 features
# D. Daily consumption + time shares + peak_to_average
#
# K = 2..6 is tested for every feature set.
# ============================================================

FILE = "data/CEEW - Smart meter data Bareilly 2020.csv"
MIN_COVERAGE = 80
EXPECTED_READINGS = 366 * 24 * 20
OUTPUT_DIR = "model_v2_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ORIGINAL_FEATURES = [
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
    "weekday_weekend_difference"
]

TIME_FEATURES = [
    "morning_share",
    "afternoon_share",
    "evening_share",
    "night_share"
]

FEATURE_SETS = {
    "A_original_11": ORIGINAL_FEATURES,
    "B_original_plus_time": ORIGINAL_FEATURES + TIME_FEATURES,
    "C_user_facing_5": [
        "avg_daily_kwh",
        "morning_share",
        "afternoon_share",
        "evening_share",
        "night_share"
    ],
    "D_daily_time_peak": [
        "avg_daily_kwh",
        "morning_share",
        "afternoon_share",
        "evening_share",
        "night_share",
        "peak_to_average"
    ]
}

print("=" * 60)
print("MODEL EXPERIMENT V2")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")
df = pd.read_csv(FILE)

print("Rows:", len(df))
print("Meters:", df["meter"].nunique())

df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])
df["t_kWh"] = pd.to_numeric(df["t_kWh"], errors="coerce").fillna(0)

df["hour"] = df["x_Timestamp"].dt.hour
df["date"] = df["x_Timestamp"].dt.date
df["dayofweek"] = df["x_Timestamp"].dt.dayofweek

# ============================================================
# COVERAGE FILTER
# ============================================================

meter_counts = df.groupby("meter").size()
coverage = (meter_counts / EXPECTED_READINGS) * 100

valid_meters = coverage[coverage >= MIN_COVERAGE].index

df = df[df["meter"].isin(valid_meters)].copy()

print("\nRetained meters:", len(valid_meters))
print("Rows after coverage filtering:", len(df))

# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating features...")

features = []

for meter, group in df.groupby("meter"):

    consumption = group["t_kWh"]

    mean_consumption = consumption.mean()
    max_consumption = consumption.max()
    std_consumption = consumption.std()
    median_consumption = consumption.median()

    total_consumption = consumption.sum()

    if mean_consumption > 0:
        peak_to_average = max_consumption / mean_consumption
    else:
        peak_to_average = 0

    # --------------------------------------------------------
    # Original time averages
    # --------------------------------------------------------

    daytime = group[
        (group["hour"] >= 6) &
        (group["hour"] < 18)
    ]["t_kWh"]

    evening = group[
        (group["hour"] >= 18) &
        (group["hour"] < 22)
    ]["t_kWh"]

    night = group[
        (group["hour"] >= 22) |
        (group["hour"] < 6)
    ]["t_kWh"]

    daytime_avg = daytime.mean()
    evening_avg = evening.mean()
    night_avg = night.mean()

    weekday = group[group["dayofweek"] < 5]["t_kWh"]
    weekend = group[group["dayofweek"] >= 5]["t_kWh"]

    weekday_avg = weekday.mean()
    weekend_avg = weekend.mean()

    weekday_weekend_difference = abs(
        weekday_avg - weekend_avg
    )

    # --------------------------------------------------------
    # New four time-period shares
    # --------------------------------------------------------

    morning_energy = group[
        (group["hour"] >= 6) &
        (group["hour"] < 12)
    ]["t_kWh"].sum()

    afternoon_energy = group[
        (group["hour"] >= 12) &
        (group["hour"] < 18)
    ]["t_kWh"].sum()

    evening_energy = group[
        (group["hour"] >= 18) &
        (group["hour"] < 22)
    ]["t_kWh"].sum()

    night_energy = group[
        (group["hour"] >= 22) |
        (group["hour"] < 6)
    ]["t_kWh"].sum()

    total_period_energy = (
        morning_energy +
        afternoon_energy +
        evening_energy +
        night_energy
    )

    if total_period_energy > 0:
        morning_share = morning_energy / total_period_energy
        afternoon_share = afternoon_energy / total_period_energy
        evening_share = evening_energy / total_period_energy
        night_share = night_energy / total_period_energy
    else:
        morning_share = 0
        afternoon_share = 0
        evening_share = 0
        night_share = 0

    # --------------------------------------------------------
    # Direct daily-energy calculation
    # --------------------------------------------------------
    #
    # Each reading represents a 3-minute interval.
    # A complete day has 480 intervals.
    #
    # For each observed day:
    #   daily total = sum of observed interval energy
    #
    # To avoid making a partially observed day look like a
    # low-consumption day, normalize its total by its coverage
    # fraction when at least 80% of the day's expected readings
    # are present.
    #
    # Days with less than 80% daily coverage are excluded from
    # this daily-average calculation.
    # --------------------------------------------------------

    group_daily = (
        group.groupby("date")
        .agg(
            daily_kwh=("t_kWh", "sum"),
            readings=("t_kWh", "size")
        )
    )

    group_daily["coverage_fraction"] = (
        group_daily["readings"] / 480
    )

    reliable_days = group_daily[
        group_daily["coverage_fraction"] >= 0.80
    ].copy()

    if len(reliable_days) > 0:
        reliable_days["normalized_daily_kwh"] = (
            reliable_days["daily_kwh"] /
            reliable_days["coverage_fraction"]
        )

        avg_daily_kwh = (
            reliable_days["normalized_daily_kwh"].mean()
        )
    else:
        avg_daily_kwh = mean_consumption * 480

    features.append({
        "meter": meter,

        "mean_consumption": mean_consumption,
        "max_consumption": max_consumption,
        "std_consumption": std_consumption,
        "median_consumption": median_consumption,
        "total_consumption": total_consumption,
        "peak_to_average": peak_to_average,

        "daytime_avg": daytime_avg,
        "evening_avg": evening_avg,
        "night_avg": night_avg,

        "weekday_avg": weekday_avg,
        "weekend_avg": weekend_avg,
        "weekday_weekend_difference":
            weekday_weekend_difference,

        "avg_daily_kwh": avg_daily_kwh,

        "morning_share": morning_share,
        "afternoon_share": afternoon_share,
        "evening_share": evening_share,
        "night_share": night_share
    })

feature_df = pd.DataFrame(features).sort_values("meter")

feature_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "consumer_features_v2.csv"
    ),
    index=False
)

print("\nFeature dataset created.")
print("Shape:", feature_df.shape)

# ============================================================
# SHOW USER-FACING FEATURES
# ============================================================

print("\nUser-facing profile features:")
print(
    feature_df[
        [
            "meter",
            "avg_daily_kwh",
            "morning_share",
            "afternoon_share",
            "evening_share",
            "night_share"
        ]
    ].to_string(index=False)
)

# ============================================================
# K-MEANS EXPERIMENT
# ============================================================

all_results = []

print("\n" + "=" * 60)
print("K-MEANS COMPARISON")
print("=" * 60)

for set_name, feature_columns in FEATURE_SETS.items():

    X = feature_df[feature_columns].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--------------------------------------------")
    print(set_name)
    print("Features:", feature_columns)
    print("--------------------------------------------")

    for k in range(2, 7):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(
            X_scaled,
            labels
        )

        db = davies_bouldin_score(
            X_scaled,
            labels
        )

        sizes = np.bincount(labels)

        result = {
            "feature_set": set_name,
            "k": k,
            "silhouette_score": silhouette,
            "davies_bouldin_score": db,
            "min_cluster_size": sizes.min(),
            "max_cluster_size": sizes.max(),
            "cluster_sizes": str(sizes.tolist())
        }

        all_results.append(result)

        print(
            f"K={k} | "
            f"Silhouette={silhouette:.4f} | "
            f"DB={db:.4f} | "
            f"Sizes={sizes.tolist()}"
        )

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BEST METRIC RESULTS")
print("=" * 60)

best_silhouette = (
    results_df
    .sort_values("silhouette_score", ascending=False)
    .head(10)
)

print(
    best_silhouette[
        [
            "feature_set",
            "k",
            "silhouette_score",
            "davies_bouldin_score",
            "cluster_sizes"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 60)
print("K=3 COMPARISON")
print("=" * 60)

k3 = results_df[results_df["k"] == 3].copy()

print(
    k3[
        [
            "feature_set",
            "silhouette_score",
            "davies_bouldin_score",
            "cluster_sizes"
        ]
    ].to_string(index=False)
)

print("\nFiles created:")
print("-", OUTPUT_DIR + "/consumer_features_v2.csv")
print("-", OUTPUT_DIR + "/model_comparison.csv")

print("\nIMPORTANT:")
print("Do not use a final model yet.")
print("Use these results to choose the feature set and K.")
