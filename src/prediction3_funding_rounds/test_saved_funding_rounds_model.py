import joblib
import pandas as pd
import numpy as np


print("=" * 60)
print("PREDICTION 3 - TESTING SAVED MODEL")
print("=" * 60)


# --------------------------------------------------
# 1. Load final model package
# --------------------------------------------------

model_path = "models/funding_rounds_final.pkl"

model_data = joblib.load(model_path)

model = model_data["model"]
preprocessor = model_data["preprocessor"]
features = model_data["features"]


print("\nFinal model loaded successfully.")


# --------------------------------------------------
# 2. Create sample startup
# --------------------------------------------------

sample_startup = {
    "category_main": "Software",
    "market": "Software",
    "funding_total_usd": 5000000,
    "country_code": "USA",
    "state_code": "CA",
    "region": "California",
    "city": "San Francisco",
    "founded_year": 2018
}


sample_df = pd.DataFrame(
    [sample_startup]
)[features]


print("\nSample Startup:")
print(sample_df.to_string(index=False))


# --------------------------------------------------
# 3. Preprocess sample
# --------------------------------------------------

sample_processed = preprocessor.transform(
    sample_df
)


# --------------------------------------------------
# 4. Make prediction
# --------------------------------------------------

prediction = model.predict(
    sample_processed
)[0]


# Funding rounds cannot be negative
prediction = max(prediction, 0)


# --------------------------------------------------
# 5. Display prediction
# --------------------------------------------------

print("\n" + "-" * 60)
print("PREDICTION RESULT")
print("-" * 60)

print(
    f"Predicted Funding Rounds: {prediction:.2f}"
)

print(
    f"Approximate Funding Rounds: {round(prediction)}"
)


print("\n" + "=" * 60)
print("SAVED MODEL TEST COMPLETED")
print("=" * 60)