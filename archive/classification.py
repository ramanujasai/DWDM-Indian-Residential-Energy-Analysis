import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# LOAD CLUSTERED DATA
# ==========================================

df = pd.read_csv("consumer_clusters.csv")

print("Dataset loaded.")
print("Shape:", df.shape)


# ==========================================
# FEATURES
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

# K-Means cluster is the target
y = df["cluster"]


# ==========================================
# SHOW CLASS DISTRIBUTION
# ==========================================

print("\n=========================================")
print("TARGET CLASS DISTRIBUTION")
print("=========================================")

print(y.value_counts().sort_index())


# ==========================================
# CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# CLASSIFIERS
# ==========================================

models = {

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=3))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ])
}


# ==========================================
# RESULTS STORAGE
# ==========================================

results = []


# ==========================================
# TRAIN AND EVALUATE
# ==========================================

for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })


# ==========================================
# RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("FINAL CLASSIFICATION RESULTS")
print("=" * 60)

print(
    results_df.round(4).to_string(index=False)
)


# ==========================================
# SAVE RESULTS
# ==========================================

results_df.to_csv(
    "classification_results.csv",
    index=False
)

print("\nResults saved to:")
print("classification_results.csv")