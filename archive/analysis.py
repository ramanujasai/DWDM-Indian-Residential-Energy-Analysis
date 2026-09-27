import pandas as pd
import os

# ==============================
# SETTINGS
# ==============================

FILE = "data/CEEW - Smart meter data Bareilly 2020.csv"
OUTPUT = "consumer_features.csv"

MIN_COVERAGE = 80

# 2020 is a leap year
EXPECTED_READINGS = 366 * 24 * 20


# ==============================
# LOAD DATA
# ==============================

print("Loading dataset...")

df = pd.read_csv(FILE)

print("Dataset loaded.")
print("Rows:", len(df))
print("Meters:", df["meter"].nunique())


# ==============================
# TIMESTAMP PROCESSING
# ==============================

df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])

df["hour"] = df["x_Timestamp"].dt.hour
df["dayofweek"] = df["x_Timestamp"].dt.dayofweek


# ==============================
# CALCULATE METER COVERAGE
# ==============================

print("\nCalculating meter coverage...")

meter_counts = df.groupby("meter").size()

coverage = (meter_counts / EXPECTED_READINGS) * 100

valid_meters = coverage[coverage >= MIN_COVERAGE].index

removed_meters = coverage[coverage < MIN_COVERAGE].index

print("\nCoverage filtering:")
print("Minimum required coverage:", MIN_COVERAGE, "%")
print("Valid meters:", len(valid_meters))
print("Removed meters:", len(removed_meters))

print("\nRemoved meters:")
for meter in removed_meters:
    print(f"{meter}: {coverage[meter]:.2f}%")


# ==============================
# FILTER DATA
# ==============================

df = df[df["meter"].isin(valid_meters)].copy()

print("\nData after coverage filtering:")
print("Rows:", len(df))
print("Meters:", df["meter"].nunique())


# ==============================
# FEATURE ENGINEERING
# ==============================

print("\nCreating consumer-level features...")

features = []

for meter, group in df.groupby("meter"):

    consumption = group["t_kWh"]

    mean_consumption = consumption.mean()
    max_consumption = consumption.max()
    min_consumption = consumption.min()
    std_consumption = consumption.std()
    median_consumption = consumption.median()
    total_consumption = consumption.sum()

    # Avoid division by zero
    if mean_consumption != 0:
        peak_to_average = max_consumption / mean_consumption
    else:
        peak_to_average = 0

    # Time-based consumption shares
    # These features are easier to explain to a normal user:
    # morning = 06:00-12:00
    # afternoon = 12:00-18:00
    # evening = 18:00-22:00
    # night = 22:00-06:00

    morning = group[
        (group["hour"] >= 6) &
        (group["hour"] < 12)
    ]["t_kWh"]

    afternoon = group[
        (group["hour"] >= 12) &
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

    total_time_consumption = (
        morning.sum() +
        afternoon.sum() +
        evening.sum() +
        night.sum()
    )

    if total_time_consumption > 0:
        morning_share = morning.sum() / total_time_consumption
        afternoon_share = afternoon.sum() / total_time_consumption
        evening_share = evening.sum() / total_time_consumption
        night_share = night.sum() / total_time_consumption
    else:
        morning_share = 0
        afternoon_share = 0
        evening_share = 0
        night_share = 0

    # Weekday / weekend
    weekday = group[
        group["dayofweek"] < 5
    ]["t_kWh"]

    weekend = group[
        group["dayofweek"] >= 5
    ]["t_kWh"]

    weekday_avg = weekday.mean()
    weekend_avg = weekend.mean()

    weekday_weekend_difference = abs(
        weekday_avg - weekend_avg
    )

    # Store features
    features.append({
        "meter": meter,

        "mean_consumption": mean_consumption,
        "max_consumption": max_consumption,
        "min_consumption": min_consumption,
        "std_consumption": std_consumption,
        "median_consumption": median_consumption,
        "total_consumption": total_consumption,
        "peak_to_average": peak_to_average,

        "morning_share": morning_share,
        "afternoon_share": afternoon_share,
        "evening_share": evening_share,
        "night_share": night_share,

        "weekday_avg": weekday_avg,
        "weekend_avg": weekend_avg,
        "weekday_weekend_difference": weekday_weekend_difference
    })


# ==============================
# CREATE FEATURE DATAFRAME
# ==============================

feature_df = pd.DataFrame(features)

# Sort by meter name
feature_df = feature_df.sort_values("meter")

# Save
feature_df.to_csv(OUTPUT, index=False)


# ==============================
# FINAL INFORMATION
# ==============================

print("\n===================================")
print("FEATURE EXTRACTION COMPLETE")
print("===================================")

print("Output file:", OUTPUT)
print("Number of consumers:", len(feature_df))
print("Number of features:", len(feature_df.columns) - 1)

print("\nFeature columns:")
for column in feature_df.columns:
    print("-", column)

print("\nFeature dataset shape:")
print(feature_df.shape)

print("\nFirst 5 rows:")
print(feature_df.head())

print("\nCoverage of retained meters:")
print(coverage[coverage >= MIN_COVERAGE].describe())

print("\nSaved successfully!")