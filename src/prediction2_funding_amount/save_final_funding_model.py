import joblib

# Load trained model and preprocessor
model = joblib.load(
    "models/funding_amount_xgboost_location_log.pkl"
)

preprocessor = joblib.load(
    "models/funding_amount_location_preprocessor.pkl"
)

# Combine them into one package
model_data = {
    "model": model,
    "preprocessor": preprocessor,
    "target_transform": "log1p",
    "features": [
        "category_main",
        "market",
        "country_code",
        "state_code",
        "region",
        "city",
        "founded_year",
        "funding_rounds"
    ]
}

# Save final package
joblib.dump(
    model_data,
    "models/funding_amount_final.pkl"
)

print("=" * 60)
print("FINAL FUNDING AMOUNT MODEL SAVED")
print("=" * 60)
print()
print("File: models/funding_amount_final.pkl")
print()
print("Model + Preprocessor packaged successfully.")
print("=" * 60)