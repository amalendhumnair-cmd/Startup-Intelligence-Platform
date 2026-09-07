import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)
print("PREDICTION 3 - LINEAR REGRESSION")
print("=" * 60)


# 1. Load prepared data
df = pd.read_csv("datasets/funding_rounds_prepared.csv")


# 2. Features and target
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


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 4. Load preprocessor
preprocessor = joblib.load(
    "models/funding_rounds_preprocessor.pkl"
)


# 5. Preprocess
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# 6. Create model
model = LinearRegression()


# 7. Train
print("\nTraining Linear Regression...")

model.fit(
    X_train_processed,
    y_train
)


# 8. Predict
y_pred = model.predict(X_test_processed)

# Funding rounds cannot be negative
y_pred = np.maximum(y_pred, 0)


# 9. Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# 10. Display results
print("\nLinear Regression Results")
print("-" * 40)

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


# 11. Show sample predictions
print("\nSample Predictions")
print("-" * 40)

comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})

print(comparison)


print("\n" + "=" * 60)
print("LINEAR REGRESSION TEST COMPLETED")
print("=" * 60)