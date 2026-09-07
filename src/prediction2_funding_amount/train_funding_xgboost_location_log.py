import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("FUNDING AMOUNT - LOCATION LOG-TARGET XGBOOST")
print("=" * 60)

# Load preprocessed location-feature data
X_train = joblib.load(
    "datasets/funding_amount_location_X_train.pkl"
)

X_test = joblib.load(
    "datasets/funding_amount_location_X_test.pkl"
)

y_train = joblib.load(
    "datasets/funding_amount_location_y_train.pkl"
)

y_test = joblib.load(
    "datasets/funding_amount_location_y_test.pkl"
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# Log transform the target
y_train_log = np.log1p(y_train)

print("\nTraining XGBoost with log-transformed target...")

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_log)

print("Training completed.")

# Predict in log scale
y_pred_log = model.predict(X_test)

# Convert predictions back to original dollar scale
y_pred = np.expm1(y_pred_log)

# Prevent negative predictions
y_pred = np.maximum(y_pred, 0)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("LOCATION LOG-TARGET XGBOOST RESULTS")
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
    "models/funding_amount_xgboost_location_log.pkl"
)

print("\nLocation Log-XGBoost model saved to:")
print("models/funding_amount_xgboost_location_log.pkl")

print("\n" + "=" * 60)
print("LOCATION LOG-TARGET XGBOOST COMPLETED")
print("=" * 60)