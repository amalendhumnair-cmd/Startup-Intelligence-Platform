import joblib
import numpy as np

print("=" * 60)
print("FUNDING AMOUNT - FINAL MODEL TEST")
print("=" * 60)

# Load saved model
model = joblib.load(
    "models/funding_amount_xgboost_location_log.pkl"
)

# Load saved preprocessor
preprocessor = joblib.load(
    "models/funding_amount_location_preprocessor.pkl"
)

# New startup sample
startup = {
    "category_main": "Software",
    "market": "Software",
    "country_code": "USA",
    "state_code": "CA",
    "region": "California",
    "city": "San Francisco",
    "founded_year": 2018,
    "funding_rounds": 3
}

print("\nStartup Input:")
print("-" * 60)

for key, value in startup.items():
    print(f"{key}: {value}")

# Convert input to DataFrame
import pandas as pd

input_data = pd.DataFrame([startup])

# Preprocess input
input_processed = preprocessor.transform(input_data)

# Predict log funding
predicted_log = model.predict(input_processed)

# Convert back to dollars
predicted_funding = np.expm1(predicted_log[0])

# Prevent negative value
predicted_funding = max(predicted_funding, 0)

print("\n" + "=" * 60)
print("FINAL FUNDING PREDICTION")
print("=" * 60)

print(f"\nPredicted Funding Amount: ${predicted_funding:,.2f}")

print("\n" + "=" * 60)
print("FINAL MODEL TEST COMPLETED")
print("=" * 60)