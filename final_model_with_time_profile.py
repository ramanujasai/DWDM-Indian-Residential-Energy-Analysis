import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ============================================================
# FINAL CLUSTERING + TIME-OF-DAY PROFILE
# ============================================================
#
# Core model:
#   K-Means K=3 using the original 11 clustering features.
#
# Separate profile/recommendation layer:
#   morning_share       06:00-12:00
#   afternoon_share     12:00-18:00
#   evening_share       18:00-22:00
#   night_share         22:00-06:00
#
# The time features are NOT used to train K-Means.
# They are used to explain WHEN a household consumes most
# of its electricity and support time-specific recommendations.
# ============================================================

FEATURE_FILE = "model_v2_results/consumer_features_v2.csv"
DATA_FILE = "data/CEEW - Smart meter data Bareilly 2020.csv"

MODEL_DIR = "models"
RESULT_DIR = "final_model_results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Original 11 features used by the project's K-Means model.
CLUSTER_FEATURES = [
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

# ============================================================
# LOAD CONSUMER FEATURES
# ============================================================

print("Loading consumer features...")

features_df = pd.read_csv(FEATURE_FILE)

missing = [
    col for col in CLUSTER_FEATURES
    if col not in features_df.columns
]

if missing:
    raise ValueError(
        f"Missing clustering features: {missing}"
    )

print("Consumers:", len(features_df))

# ============================================================
# TRAIN FINAL K-MEANS
# ============================================================

print("\nTraining final K-Means model...")

X = features_df[CLUSTER_FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

features_df["cluster"] = kmeans.fit_predict(X_scaled)

print("K-Means trained.")
print("\nCluster sizes:")
print(
    features_df["cluster"]
    .value_counts()
    .sort_index()
)

# ============================================================
# LOAD RAW DATA FOR TIME-OF-DAY PROFILE
# ============================================================

print("\nLoading raw data for time-of-day profiles...")

raw = pd.read_csv(
    DATA_FILE,
    usecols=["meter", "x_Timestamp", "t_kWh"]
)

raw["x_Timestamp"] = pd.to_datetime(
    raw["x_Timestamp"]
)

raw["t_kWh"] = pd.to_numeric(
    raw["t_kWh"],
    errors="coerce"
).fillna(0)

raw["hour"] = raw["x_Timestamp"].dt.hour

# Only use households present in the final clustering dataset.
raw = raw[
    raw["meter"].isin(features_df["meter"])
].copy()

# ============================================================
# TIME-OF-DAY SHARES
# ============================================================

print("Calculating time-of-day consumption shares...")

def period_share(group, condition):
    energy = group.loc[condition(group["hour"]), "t_kWh"].sum()
    total = group["t_kWh"].sum()

    if total <= 0:
        return 0.0

    return energy / total


time_profiles = []

for meter, group in raw.groupby("meter"):

    morning_share = period_share(
        group,
        lambda h: (h >= 6) & (h < 12)
    )

    afternoon_share = period_share(
        group,
        lambda h: (h >= 12) & (h < 18)
    )

    evening_share = period_share(
        group,
        lambda h: (h >= 18) & (h < 22)
    )

    night_share = period_share(
        group,
        lambda h: (h >= 22) | (h < 6)
    )

    shares = {
        "Morning": morning_share,
        "Afternoon": afternoon_share,
        "Evening": evening_share,
        "Night": night_share
    }

    dominant_period = max(
        shares,
        key=shares.get
    )

    time_profiles.append({
        "meter": meter,
        "morning_share": morning_share,
        "afternoon_share": afternoon_share,
        "evening_share": evening_share,
        "night_share": night_share,
        "dominant_period": dominant_period
    })

time_df = pd.DataFrame(time_profiles)

# ============================================================
# COMBINE CLUSTER + TIME PROFILE
# ============================================================

# The V2 feature file already contains the four time-share
# columns. Remove them before merging the freshly calculated
# raw-data values so pandas does not create _x/_y duplicates.
profile_df = features_df.drop(
    columns=[
        "morning_share",
        "afternoon_share",
        "evening_share",
        "night_share"
    ],
    errors="ignore"
).merge(
    time_df,
    on="meter",
    how="left"
)

profile_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "household_profiles.csv"
    ),
    index=False
)

# ============================================================
# CLUSTER PROFILES
# ============================================================

cluster_profiles = (
    profile_df
    .groupby("cluster")
    .agg(
        households=("meter", "count"),

        mean_consumption=("mean_consumption", "mean"),
        max_consumption=("max_consumption", "mean"),
        median_consumption=("median_consumption", "mean"),
        peak_to_average=("peak_to_average", "mean"),

        morning_share=("morning_share", "mean"),
        afternoon_share=("afternoon_share", "mean"),
        evening_share=("evening_share", "mean"),
        night_share=("night_share", "mean")
    )
    .reset_index()
)

# Determine dominant period of each cluster.
period_columns = {
    "Morning": "morning_share",
    "Afternoon": "afternoon_share",
    "Evening": "evening_share",
    "Night": "night_share"
}

cluster_dominant_periods = []

for _, row in cluster_profiles.iterrows():

    dominant = max(
        period_columns,
        key=lambda name: row[period_columns[name]]
    )

    cluster_dominant_periods.append(dominant)

cluster_profiles["dominant_period"] = (
    cluster_dominant_periods
)

cluster_profiles.to_csv(
    os.path.join(
        RESULT_DIR,
        "cluster_profiles.csv"
    ),
    index=False
)

# ============================================================
# SAVE MODELS
# ============================================================

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "final_kmeans_scaler.pkl"
    )
)

joblib.dump(
    kmeans,
    os.path.join(
        MODEL_DIR,
        "final_kmeans.pkl"
    )
)

# ============================================================
# RECOMMENDATION LOGIC
# ============================================================
#
# This function is intentionally based on TIME PERIOD,
# not appliance assumptions.
#
# K-Means does not tell us which appliance caused usage.
# ============================================================

RECOMMENDATIONS = {
    "Morning": (
        "Your highest electricity use occurs in the morning "
        "(6 AM-12 PM). Review flexible morning activities "
        "and avoid running several high-consumption devices "
        "at the same time where practical."
    ),

    "Afternoon": (
        "Your highest electricity use occurs in the afternoon "
        "(12 PM-6 PM). Review appliances or activities that "
        "remain active during the afternoon and switch off "
        "unnecessary loads when they are not needed."
    ),

    "Evening": (
        "Your highest electricity use occurs in the evening "
        "(6 PM-10 PM). Consider reducing simultaneous use of "
        "high-consumption devices and shifting flexible usage "
        "outside this period where practical."
    ),

    "Night": (
        "Your highest electricity use occurs at night "
        "(10 PM-6 AM). Check for devices that remain active "
        "overnight and switch off unnecessary loads before "
        "going to sleep."
    )
}

# Save recommendation text for later app integration.
recommendation_df = pd.DataFrame(
    [
        {
            "period": period,
            "recommendation": recommendation
        }
        for period, recommendation
        in RECOMMENDATIONS.items()
    ]
)

recommendation_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "time_recommendations.csv"
    ),
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPLETE")
print("=" * 60)

print("\nClustering features:")
for feature in CLUSTER_FEATURES:
    print("-", feature)

print("\nTime-profile features:")
for feature in TIME_FEATURES:
    print("-", feature)

print("\nCluster profiles:")
print(
    cluster_profiles.to_string(index=False)
)

print("\nHousehold profiles:")
print(
    profile_df[
        [
            "meter",
            "cluster",
            "morning_share",
            "afternoon_share",
            "evening_share",
            "night_share",
            "dominant_period"
        ]
    ].to_string(index=False)
)

print("\nSaved files:")
print("-", RESULT_DIR + "/household_profiles.csv")
print("-", RESULT_DIR + "/cluster_profiles.csv")
print("-", RESULT_DIR + "/time_recommendations.csv")
print("-", MODEL_DIR + "/final_kmeans_scaler.pkl")
print("-", MODEL_DIR + "/final_kmeans.pkl")

print("\nThe core K-Means model uses ONLY the original")
print("11 features. Time-of-day features are kept")
print("separately for profile explanation and recommendations.")
