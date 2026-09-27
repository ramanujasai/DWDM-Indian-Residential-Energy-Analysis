# Indian Residential Energy Consumption Pattern Analysis

**DWDM Lab Project • 23CS3551**

## 1. Project Overview

This project analyzes electricity consumption patterns of Indian residential consumers using smart-meter data. Consumer-level statistical and time-based features are extracted from smart-meter readings and used with clustering techniques to identify groups of consumers with similar consumption behaviour. A saved K-Means model is then used to analyze a new consumer dataset.

## 2. Problem Statement

Residential electricity consumers can exhibit different usage patterns even when they are measured using the same smart-meter system. The objective of this project is to group consumers according to their electricity consumption characteristics and apply the learned grouping to a new household's smart-meter readings.

## 3. Dataset

The project uses the **CEEW Smart Meter Data from Bareilly, India, 2020**.

- Period: January 1, 2020 to December 31, 2020
- Original meters: 46
- Raw readings: 6,627,360
- Sampling interval: 3 minutes
- Primary consumption field: `t_kWh`
- Timestamp field: `x_Timestamp`
- Meter identifier: `meter`
- Additional recorded fields include average voltage, average current and frequency.

Dataset references:

- Kaggle: https://www.kaggle.com/datasets/pythonafroz/electricity-smart-meter-data-from-india
- Harvard Dataverse: https://doi.org/10.7910/DVN/GOCHJH

## 4. Data Preprocessing

The raw data was processed at the consumer level.

1. Timestamps were converted to datetime values.
2. Invalid timestamps and invalid consumption values were removed.
3. Negative consumption values were excluded.
4. Annual meter coverage was calculated against the expected number of 3-minute readings in the leap year 2020.
5. A minimum coverage threshold of 80% was used for annual consumer-level analysis.
6. 32 of the original 46 meters satisfied this coverage requirement.

The coverage filter was used to reduce the influence of meters with very incomplete annual records or extremely sparse consumption data.

## 5. Feature Engineering

Eleven consumer-level features were used by the final K-Means model:

| Feature | Meaning |
|---|---|
| `mean_consumption` | Average consumption |
| `max_consumption` | Maximum observed consumption |
| `std_consumption` | Consumption variability |
| `median_consumption` | Median consumption |
| `peak_to_average` | Peak relative to average consumption |
| `daytime_avg` | Average usage from 06:00–18:00 |
| `evening_avg` | Average usage from 18:00–22:00 |
| `night_avg` | Average usage from 22:00–06:00 |
| `weekday_avg` | Average weekday usage |
| `weekend_avg` | Average weekend usage |
| `weekday_weekend_difference` | Absolute difference between weekday and weekend averages |

For the final model, the standard deviation uses the same sample-standard-deviation convention as the training feature-generation pipeline.

## 6. Scaling

Before clustering, the 11 features are standardized using `StandardScaler`.

For a feature value `x`, the transformation is:

`z = (x - mean) / standard deviation`

The scaler fitted during training is saved and reused for new consumer data. A new scaler is not fitted for each user.

## 7. Clustering

K-Means clustering was evaluated for multiple values of K. The final application uses **K = 3**.

The three discovered groups contain:

- Cluster 0: 11 consumers
- Cluster 1: 6 consumers
- Cluster 2: 15 consumers

The selected K was examined using the elbow method, silhouette score and Davies-Bouldin index.

Final K-Means silhouette score: **0.4197**.

## 8. Additional Clustering Experiments

Two additional clustering methods were examined during project development:

- Hierarchical Agglomerative Clustering (HAC)
- Gaussian Mixture Model (GMM)

The GMM produced the same consumer grouping as K-Means up to cluster-label permutation in the evaluated experiment. HAC produced a related but different grouping.

These experiments are documented for academic evaluation but are not required during normal application use.

## 9. Classification Experiments

The discovered K-Means cluster labels were also used as target labels for classification experiments using:

- K-Nearest Neighbors
- Decision Tree
- Support Vector Machine
- Logistic Regression

These classifiers were used to examine how well the discovered cluster structure could be reproduced from the engineered feature space.

Important: the K-Means cluster labels are **derived labels**, not independently observed household categories. Therefore classification accuracy should not be interpreted as real-world ground-truth classification accuracy.

## 10. New Consumer Analysis Pipeline

The application does not retrain the model every time a user uploads a CSV.

```text
User Smart-Meter CSV
        ↓
Data validation
        ↓
Feature engineering
        ↓
Saved StandardScaler
        ↓
Saved K-Means model
        ↓
Predicted cluster
        ↓
Time-of-day profile
        ↓
Consumption observations
```

Required input columns:

```text
meter
x_Timestamp
t_kWh
```

The user can upload a smart-meter CSV through **Consumer Analysis**. The application uses the saved scaler and K-Means model and does not retrain the model during normal prediction.

## 11. Time-of-Day Profile

For user-facing analysis, consumption is additionally divided into four periods:

- Morning: 06:00–12:00
- Afternoon: 12:00–18:00
- Evening: 18:00–22:00
- Night: 22:00–06:00

The period with the largest share is reported as the consumer's dominant consumption period.

## 12. Recommendations

Recommendations are generated from the measured consumer profile and its corresponding cluster profile. They focus on areas such as unusually high peak behaviour, dominant usage periods and differences from similar consumers.

The application does **not** identify individual appliances because the available smart-meter consumption signal does not contain appliance-level labels.

## 13. Model Files

The application uses the saved final model files:

```text
models/final_kmeans_scaler.pkl
models/final_kmeans.pkl
```

The original 350 MB training dataset is therefore not required for normal prediction of a new consumer.

## 14. Application Structure

The dashboard is intentionally kept focused on the current consumer. Detailed methodology and experimental results are documented here instead of filling the user-facing dashboard with training-dataset statistics.

Main application sections:

- **Dashboard**: current consumer result and profile
- **Consumer Analysis**: upload and analyze a consumer CSV
- **Technical Analysis**: compact model information and evaluation
- **About Project**: project purpose and limitations

## 15. Running the Application

From the project directory:

```bash
python -m streamlit run app.py
```

The project directory should contain `app.py` and the saved `models` directory.

## 16. Project Reproducibility

The repository contains the files required for the final application and its academic documentation:

- `app.py` - Streamlit application
- `app.py` - Streamlit application
- `models/` - saved scaler and final K-Means model used by the application
- `results/pca_clusters.png` - PCA visualization shown in Technical Analysis
- `Sample data/` - sample CSV files for application demonstration
- `docs/` - project documentation and presentation materials

The `.venv` directory is a local Python environment and is not required to be committed to source control.

## 17. Research Reference

Methodological reference:

> Categorization of Indian residential consumers electrical energy consumption pattern using clustering and classification techniques.

Energy, Volume 289, Article 129992, 2024.

DOI: https://doi.org/10.1016/j.energy.2023.129992

This paper is used as methodological reference. The project dataset is the CEEW Bareilly 2020 smart-meter dataset and is not the same dataset used in that paper.

## 18. Project Limitation

The model discovers statistical consumption patterns from the available smart-meter data. Cluster membership should therefore be interpreted as a data-driven consumption pattern rather than a fixed behavioural or appliance category.
