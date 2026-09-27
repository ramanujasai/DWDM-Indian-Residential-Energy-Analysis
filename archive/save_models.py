import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# ==========================================
# CONFIGURATION
# ==========================================

DATA_FILE = "consumer_clusters.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# These are the features used by our ML models
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
    "weekday_weekend_difference"
]


# ==========================================
# LOAD DATA
# ==========================================

print("Loading final cluster dataset...")

df = pd.read_csv(DATA_FILE)

X = df[FEATURES]
y = df["cluster"]


print("Consumers:", len(df))
print("Features:", len(FEATURES))


# ==========================================
# STANDARDIZATION
# ==========================================

print("\nTraining feature scaler...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# K-MEANS
# ==========================================

print("Training K-Means...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)


# ==========================================
# CLASSIFICATION MODELS
# ==========================================

print("Training classification models...")


knn = KNeighborsClassifier(
    n_neighbors=5
)

decision_tree = DecisionTreeClassifier(
    random_state=42
)

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

logistic = LogisticRegression(
    max_iter=1000,
    random_state=42
)


knn.fit(X_scaled, y)

decision_tree.fit(X_scaled, y)

svm.fit(X_scaled, y)

logistic.fit(X_scaled, y)


# ==========================================
# SAVE MODELS
# ==========================================

print("\nSaving models...")


joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "scaler.pkl")
)

joblib.dump(
    kmeans,
    os.path.join(MODEL_DIR, "kmeans.pkl")
)

joblib.dump(
    knn,
    os.path.join(MODEL_DIR, "knn.pkl")
)

joblib.dump(
    decision_tree,
    os.path.join(MODEL_DIR, "decision_tree.pkl")
)

joblib.dump(
    svm,
    os.path.join(MODEL_DIR, "svm.pkl")
)

joblib.dump(
    logistic,
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)


# ==========================================
# COMPLETE
# ==========================================

print("\n=========================================")
print("MODEL SAVING COMPLETE")
print("=========================================")

print("\nSaved files:")

print("models/scaler.pkl")
print("models/kmeans.pkl")
print("models/knn.pkl")
print("models/decision_tree.pkl")
print("models/svm.pkl")
print("models/logistic_regression.pkl")