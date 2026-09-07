import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)
print("PREDICTION 3 - RANDOM FOREST")
print("=" * 60)


# --------------------------------------------------
# 1. Load prepared data
# --------------------------------------------------

df = pd.read_csv(
    "datasets/funding_rounds_prepared.csv"
)

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

features = [
    "category_main",
    "market",
    "funding_total_usd",
    "country_code",
    "state_code",
    "region",
    "city",
    "founded_year"
]

target = "funding_rounds"

X = df[features]
y = df[target]


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 4. Load saved preprocessor
# --------------------------------------------------

preprocessor = joblib.load(
    "models/funding_rounds_preprocessor.pkl"
)


# --------------------------------------------------
# 5. Preprocess data
# --------------------------------------------------

X_train_processed = preprocessor.transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print("\nProcessed data:")
print("X_train:", X_train_processed.shape)
print("X_test :", X_test_processed.shape)


# --------------------------------------------------
# 6. Create Random Forest model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

print("\nTraining Random Forest...")

model.fit(
    X_train_processed,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

y_pred = model.predict(
    X_test_processed
)

# Funding rounds cannot be negative
y_pred = np.maximum(
    y_pred,
    0
)


# --------------------------------------------------
# 9. Calculate evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("\nRandom Forest Results")
print("-" * 40)

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


# --------------------------------------------------
# 11. Show sample predictions
# --------------------------------------------------

print("\nSample Predictions")
print("-" * 40)

comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})

print(comparison)


print("\n" + "=" * 60)
print("RANDOM FOREST TEST COMPLETED")
print("=" * 60)