import joblib
import pandas as pd


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model_data = joblib.load(
    "models/startup_success_xgboost.pkl"
)

pipeline = model_data["pipeline"]
threshold = model_data["threshold"]
features = model_data["features"]


# ============================================================
# 2. ENTER SAMPLE STARTUP INFORMATION
# ============================================================

startup = pd.DataFrame([{
    "category_main": "Software",
    "market": "Software",
    "funding_total_usd": 5000000,
    "funding_rounds": 3,
    "country_code": "USA",
    "region": "California",
    "founded_year": 2018
}])


# ============================================================
# 3. MAKE PREDICTION
# ============================================================

probability = pipeline.predict_proba(startup)[0][1]

if probability >= threshold:
    prediction = "SUCCESS"
else:
    prediction = "NOT SUCCESS"


# ============================================================
# 4. DISPLAY RESULT
# ============================================================

print("\n============================================================")
print("STARTUP SUCCESS PREDICTION")
print("============================================================")

print("\nStartup Information:")
print(startup.to_string(index=False))

print("\nSuccess Probability:",
      round(probability * 100, 2), "%")

print("Decision Threshold:",
      threshold)

print("\nFinal Prediction:",
      prediction)

print("\n============================================================")