import pandas as pd
import numpy as np

# load original dataset
df = pd.read_csv(r"C:\Users\rohin\OneDrive\Documents\Parking 19.4.26\Parking\dataset\smart_parking_realistic_10000.csv")

print("Original columns:", df.columns)

# ============================
# CLEAN
# ============================

df = df.drop_duplicates().copy()
df["occupied"] = df["occupied"].astype(int)

# ============================
# FEATURE ENGINEERING
# ============================

# duration
df["duration"] = (df["exit_hour"] - df["entry_hour"]) % 24
df.loc[df["duration"] <= 0, "duration"] = 1

# cyclic
df["entry_sin"] = np.sin(2 * np.pi * df["entry_hour"] / 24)
df["entry_cos"] = np.cos(2 * np.pi * df["entry_hour"] / 24)

df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# time patterns
df["peak_hour"] = df["entry_hour"].apply(lambda x: 1 if (7<=x<=10 or 17<=x<=20) else 0)
df["is_night"] = df["entry_hour"].apply(lambda x: 1 if (x>=22 or x<=5) else 0)
df["time_block"] = df["entry_hour"] // 3

# slot behavior
df["slot_group"] = df["slot_id"] % 5
df["slot_zone"] = df["slot_id"] % 3

# ============================
# REALISTIC NEW FEATURES
# ============================

# traffic (based on time)
df["traffic_level"] = df["entry_hour"].apply(
    lambda x: 2 if (7<=x<=10 or 17<=x<=20)
    else 1 if (10<x<17)
    else 0
)

# location importance
df["location_type"] = df["slot_id"].apply(lambda x: 1 if x % 3 == 0 else 0)

# stay behavior
df["is_long_stay"] = df["duration"].apply(lambda x: 1 if x >= 5 else 0)
df["is_short_stay"] = df["duration"].apply(lambda x: 1 if x <= 2 else 0)

# interaction features
df["rush_factor"] = df["peak_hour"] * (df["is_weekend"] + 1)
df["work_rush"] = (1 - df["is_weekend"]) * df["traffic_level"]
df["weekend_peak"] = df["is_weekend"] * df["peak_hour"]

# ============================
# REBUILD TARGET (IMPORTANT)
# ============================

# create realistic occupancy pattern
score = (
    2 * df["peak_hour"] +
    1.5 * df["traffic_level"] +
    1.2 * df["location_type"] +
    1.0 * df["is_long_stay"] +
    0.8 * df["work_rush"] +
    0.5 * df["weekend_peak"] -
    1.0 * df["is_night"] -
    0.5 * df["is_short_stay"]
)

# convert score → probability
prob = 1 / (1 + np.exp(-score + 2))

# generate new occupied column (REALISTIC)
np.random.seed(42)
df["occupied"] = (np.random.rand(len(df)) < prob).astype(int)

# ============================
# SAVE NEW DATASET
# ============================

df.to_csv("smart_parking_improved.csv", index=False)

print("✅ Improved dataset created: smart_parking_improved.csv")
print(df.head())
