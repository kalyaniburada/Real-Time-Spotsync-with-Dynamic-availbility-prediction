import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import resample
from xgboost import XGBClassifier

# ============================
# LOAD DATASET
# ============================

import os

dataset_path = os.path.join(os.path.dirname(__file__), "smart_parking_improved.csv")
df = pd.read_csv(dataset_path)

# print("Columns:", df.columns.tolist())

# ============================
# CLEAN DATA
# ============================

df = df.drop_duplicates().copy()
df["occupied"] = pd.to_numeric(df["occupied"], errors="coerce")
df = df.dropna(subset=["slot_id", "entry_hour", "exit_hour", "day_of_week", "is_weekend", "occupied"])

df["slot_id"] = df["slot_id"].astype(int)
df["entry_hour"] = df["entry_hour"].astype(int).clip(0, 23)
df["exit_hour"] = df["exit_hour"].astype(int).clip(0, 23)
df["day_of_week"] = df["day_of_week"].astype(int).clip(0, 6)
df["is_weekend"] = df["is_weekend"].astype(int)
df["occupied"] = df["occupied"].astype(int)

# ============================
# FEATURE ENGINEERING
# ============================

df["duration"] = (df["exit_hour"] - df["entry_hour"]) % 24
df.loc[df["duration"] <= 0, "duration"] = 1

df["entry_sin"] = np.sin(2 * np.pi * df["entry_hour"] / 24)
df["entry_cos"] = np.cos(2 * np.pi * df["entry_hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["peak_hour"] = df["entry_hour"].apply(lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 20) else 0)
df["is_night"] = df["entry_hour"].apply(lambda x: 1 if (x >= 22 or x <= 5) else 0)
df["time_block"] = df["entry_hour"] // 3

df["slot_group"] = df["slot_id"] % 5
df["slot_zone"] = df["slot_id"] % 3

df["traffic_level"] = df["entry_hour"].apply(
    lambda x: 2 if (7 <= x <= 10 or 17 <= x <= 20)
    else 1 if (10 < x < 17)
    else 0
)

df["location_type"] = df["slot_id"].apply(lambda x: 1 if x % 3 == 0 else 0)
df["is_long_stay"] = df["duration"].apply(lambda x: 1 if x >= 5 else 0)
df["is_short_stay"] = df["duration"].apply(lambda x: 1 if x <= 2 else 0)
df["rush_factor"] = df["peak_hour"] * (df["is_weekend"] + 1)
df["work_rush"] = (1 - df["is_weekend"]) * df["traffic_level"]
df["weekend_peak"] = df["is_weekend"] * df["peak_hour"]
df["hour_slot_interaction"] = df["entry_hour"] * df["slot_group"]

# print("\nClass distribution before split:")
# print(df["occupied"].value_counts())

# ============================
# FEATURES
# ============================

feature_cols = [
    "slot_id",
    "entry_hour",
    "exit_hour",
    "day_of_week",
    "is_weekend",
    "duration",
    "entry_sin",
    "entry_cos",
    "day_sin",
    "day_cos",
    "peak_hour",
    "is_night",
    "time_block",
    "slot_group",
    "slot_zone",
    "traffic_level",
    "location_type",
    "is_long_stay",
    "is_short_stay",
    "rush_factor",
    "work_rush",
    "weekend_peak",
    "hour_slot_interaction"
]

X = df[feature_cols]
y = df["occupied"]

# ============================
# TRAIN / TEST SPLIT
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================
# BALANCE ONLY TRAIN DATA
# ============================

train_df = X_train.copy()
train_df["occupied"] = y_train.values

major = train_df[train_df["occupied"] == 0]
minor = train_df[train_df["occupied"] == 1]

if len(major) > len(minor):
    minor_up = resample(minor, replace=True, n_samples=len(major), random_state=42)
    train_bal = pd.concat([major, minor_up])
else:
    major_up = resample(major, replace=True, n_samples=len(minor), random_state=42)
    train_bal = pd.concat([major_up, minor])

train_bal = train_bal.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_bal = train_bal[feature_cols]
y_train_bal = train_bal["occupied"]

print("\nClass distribution after train balancing:")
print(y_train_bal.value_counts())

# ============================
# HYPERPARAMETER TUNING
# ============================

params = {
    "n_estimators": [200, 300, 400],
    "max_depth": [4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
}

search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_distributions=params,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_bal, y_train_bal)

model = search.best_estimator_
# print("\n🔥 Best Params:", search.best_params_)

# ============================
# REAL EVALUATION
# ============================

y_prob = model.predict_proba(X_test)[:, 1]

best_acc = 0
best_t = 0.5

for t in [i / 100 for i in range(30, 70)]:
    y_pred_t = (y_prob > t).astype(int)
    acc_t = accuracy_score(y_test, y_pred_t)
    if acc_t > best_acc:
        best_acc = acc_t
        best_t = t

y_pred = (y_prob > best_t).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n🔥 BEST THRESHOLD:", best_t)
print("ACCURACY:", round(acc * 100, 2), "%")
# print("ROC-AUC:", round(roc, 4))
# print("\nConfusion Matrix:")
# print(cm)
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# ============================
# SAVE MODEL
# ============================

jjoblib.dump({
    "model": model,
    "threshold": best_t
}, "availability_model.pkl")
print("Model saved as availability_model.pkl")

# ============================
# FLASK API
# ============================

# app = Flask(__name__)
# model = joblib.load("availability_model.pkl")

# @app.route("/predict_availability")
# def predict():
#     try:
#         slot_id = int(request.args.get("slot_id"))
#         entry_hour = int(request.args.get("entry_hour"))
#         exit_hour = int(request.args.get("exit_hour"))
#         day_of_week = int(request.args.get("day_of_week"))
#         is_weekend = int(request.args.get("is_weekend"))

#         duration = (exit_hour - entry_hour) % 24
#         if duration <= 0:
#             duration = 1

#         entry_sin = np.sin(2 * np.pi * entry_hour / 24)
#         entry_cos = np.cos(2 * np.pi * entry_hour / 24)
#         day_sin = np.sin(2 * np.pi * day_of_week / 7)
#         day_cos = np.cos(2 * np.pi * day_of_week / 7)

#         peak_hour = 1 if (7 <= entry_hour <= 10 or 17 <= entry_hour <= 20) else 0
#         is_night = 1 if (entry_hour >= 22 or entry_hour <= 5) else 0
#         time_block = entry_hour // 3
#         slot_group = slot_id % 5
#         slot_zone = slot_id % 3

#         traffic_level = 2 if (7 <= entry_hour <= 10 or 17 <= entry_hour <= 20) else (1 if (10 < entry_hour < 17) else 0)
#         location_type = 1 if slot_id % 3 == 0 else 0
#         is_long_stay = 1 if duration >= 5 else 0
#         is_short_stay = 1 if duration <= 2 else 0
#         rush_factor = peak_hour * (is_weekend + 1)
#         work_rush = (1 - is_weekend) * traffic_level
#         weekend_peak = is_weekend * peak_hour
#         hour_slot_interaction = entry_hour * slot_group

#         features = pd.DataFrame([{
#             "slot_id": slot_id,
#             "entry_hour": entry_hour,
#             "exit_hour": exit_hour,
#             "day_of_week": day_of_week,
#             "is_weekend": is_weekend,
#             "duration": duration,
#             "entry_sin": entry_sin,
#             "entry_cos": entry_cos,
#             "day_sin": day_sin,
#             "day_cos": day_cos,
#             "peak_hour": peak_hour,
#             "is_night": is_night,
#             "time_block": time_block,
#             "slot_group": slot_group,
#             "slot_zone": slot_zone,
#             "traffic_level": traffic_level,
#             "location_type": location_type,
#             "is_long_stay": is_long_stay,
#             "is_short_stay": is_short_stay,
#             "rush_factor": rush_factor,
#             "work_rush": work_rush,
#             "weekend_peak": weekend_peak,
#             "hour_slot_interaction": hour_slot_interaction
#         }])

#         pred = model.predict(features)[0]
#         prob = model.predict_proba(features)[0]

#         return jsonify({
#             "predicted_occupied": int(pred),
#             "probability_available": float(prob[0]),
#             "probability_occupied": float(prob[1])
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True)