import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("FUNDING AMOUNT - LOG-TARGET RANDOM FOREST")
print("=" * 60)

# Load preprocessed data
X_train = joblib.load("datasets/funding_amount_X_train.pkl")
X_test = joblib.load("datasets/funding_amount_X_test.pkl")
y_train = joblib.load("datasets/funding_amount_y_train.pkl")
y_test = joblib.load("datasets/funding_amount_y_test.pkl")

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# Log transform target
y_train_log = np.log1p(y_train)

print("\nTraining Random Forest with log-transformed target...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_log)

print("Training completed.")

# Predict in log scale
y_pred_log = model.predict(X_test)

# Convert back to original dollar scale
y_pred = np.expm1(y_pred_log)

# Prevent negative predictions
y_pred = np.maximum(y_pred, 0)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("LOG-TARGET RANDOM FOREST RESULTS")
print("=" * 60)

print(f"\nMAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")

print("\nSample Predictions:")
print("=" * 60)

results = pd.DataFrame({
    "Actual Funding": y_test.values[:10],
    "Predicted Funding": y_pred[:10]
})

print(results.to_string(index=False))

# Save model
joblib.dump(
    model,
    "models/funding_amount_random_forest_log.pkl"
)

print("\nLog-target Random Forest model saved to:")
print("models/funding_amount_random_forest_log.pkl")

print("\n" + "=" * 60)
print("LOG-TARGET RANDOM FOREST COMPLETED")
print("=" * 60)