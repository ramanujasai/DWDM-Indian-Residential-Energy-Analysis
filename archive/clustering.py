import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==========================================
# LOAD FEATURE DATA
# ==========================================

df = pd.read_csv("consumer_features.csv")


# ==========================================
# FEATURES USED FOR CLUSTERING
# ==========================================

features = [
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

X = df[features]


# ==========================================
# STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# FINAL K-MEANS MODEL
# ==========================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

df["cluster"] = kmeans.fit_predict(X_scaled)


# ==========================================
# CLUSTER SIZES
# ==========================================

print("\n=========================================")
print("FINAL K-MEANS CLUSTERING")
print("=========================================")

print("\nCluster sizes:")

sizes = df["cluster"].value_counts().sort_index()

for cluster, size in sizes.items():
    print(f"Cluster {cluster}: {size} meters")


# ==========================================
# CLUSTER PROFILES
# ==========================================

print("\nCluster profiles:")

profile = df.groupby("cluster")[features].mean()

print(profile.round(4))


# ==========================================
# METER ASSIGNMENTS
# ==========================================

print("\nMeter assignments:")

for cluster in sorted(df["cluster"].unique()):

    meters = df[
        df["cluster"] == cluster
    ]["meter"].tolist()

    print(
        f"Cluster {cluster}: "
        + ", ".join(meters)
    )


# ==========================================
# SAVE FINAL RESULTS
# ==========================================

df.to_csv(
    "consumer_clusters.csv",
    index=False
)

print("\n=========================================")
print("FINAL CLUSTER DATA SAVED")
print("=========================================")

print("File: consumer_clusters.csv")
print("Shape:", df.shape)