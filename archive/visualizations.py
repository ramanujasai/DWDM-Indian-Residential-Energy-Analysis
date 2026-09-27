import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==========================================
# LOAD DATA
# ==========================================

clusters = pd.read_csv("consumer_clusters.csv")
classification = pd.read_csv("classification_results.csv")

os.makedirs("results", exist_ok=True)


# ==========================================
# 1. CLUSTER DISTRIBUTION
# ==========================================

cluster_counts = clusters["cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Number of Meters")
plt.title("Consumer Distribution Across K-Means Clusters")

plt.tight_layout()

plt.savefig(
    "results/cluster_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# 2. STANDARDIZED CLUSTER PROFILES
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

# Calculate average value for each cluster
profile = clusters.groupby("cluster")[features].mean()

# Standardize the cluster-profile values
profile_scaler = StandardScaler()

profile_scaled = pd.DataFrame(
    profile_scaler.fit_transform(profile),
    index=profile.index,
    columns=profile.columns
)

plt.figure(figsize=(13, 6))

for cluster in profile_scaled.index:

    plt.plot(
        profile_scaled.columns,
        profile_scaled.loc[cluster],
        marker="o",
        label=f"Cluster {cluster}"
    )

plt.axhline(
    y=0,
    linewidth=1
)

plt.xlabel("Consumption Features")
plt.ylabel("Standardized Average Value")
plt.title("Standardized K-Means Cluster Profiles")

plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/cluster_profiles.png",
    dpi=300
)

plt.close()


# ==========================================
# 3. PCA VISUALIZATION
# ==========================================

X = clusters[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

for cluster in sorted(clusters["cluster"].unique()):

    mask = clusters["cluster"] == cluster

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster}",
        s=70
    )

plt.xlabel(
    f"Principal Component 1 "
    f"({pca.explained_variance_ratio_[0] * 100:.1f}%)"
)

plt.ylabel(
    f"Principal Component 2 "
    f"({pca.explained_variance_ratio_[1] * 100:.1f}%)"
)

plt.title("PCA Visualization of Consumer Clusters")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/pca_clusters.png",
    dpi=300
)

plt.close()


# ==========================================
# 4. CLASSIFICATION PERFORMANCE
# ==========================================

models = classification["Model"]

accuracy = classification["Accuracy"]
f1 = classification["F1_Score"]

plt.figure(figsize=(10, 6))

x = range(len(models))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    accuracy,
    width,
    label="Accuracy"
)

plt.bar(
    [i + width / 2 for i in x],
    f1,
    width,
    label="F1 Score"
)

plt.xlabel("Classification Model")
plt.ylabel("Score")

plt.title("Classification Model Performance")

plt.xticks(
    x,
    models,
    rotation=20
)

plt.ylim(0, 1.1)

plt.legend()
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    "results/classification_performance.png",
    dpi=300
)

plt.close()


# ==========================================
# COMPLETE
# ==========================================

print("\n=========================================")
print("VISUALIZATION COMPLETE")
print("=========================================")

print("\nGenerated files:")

print("results/cluster_distribution.png")
print("results/cluster_profiles.png")
print("results/pca_clusters.png")
print("results/classification_performance.png")