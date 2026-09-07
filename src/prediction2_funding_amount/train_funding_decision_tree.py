import pandas as pd
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# ============================================================
# PREDICTION 2 - FUNDING AMOUNT
# DECISION TREE REGRESSION
# ============================================================

print("=" * 60)
print("FUNDING AMOUNT - DECISION TREE REGRESSION")
print("=" * 60)


# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================

X_train = joblib.load(
    "datasets/funding_amount_X_train.pkl"
)

X_test = joblib.load(
    "datasets/funding_amount_X_test.pkl"
)

y_train = joblib.load(
    "datasets/funding_amount_y_train.pkl"
)

y_test = joblib.load(
    "datasets/funding_amount_y_test.pkl"
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# 2. TRAIN DECISION TREE REGRESSOR
# ============================================================

print("\nTraining Decision Tree Regressor...")

model = DecisionTreeRegressor(
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 3. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 4. CALCULATE METRICS
# ============================================================

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


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE RESULTS")
print("=" * 60)

print(f"\nMAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 6. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\nSample Predictions:")
print("=" * 60)

comparison = pd.DataFrame({
    "Actual Funding": y_test.values[:10],
    "Predicted Funding": y_pred[:10]
})

print(comparison.to_string(index=False))


# ============================================================
# 7. SAVE MODEL
# ============================================================

model_path = "models/funding_amount_decision_tree.pkl"

joblib.dump(
    model,
    model_path
)

print("\nDecision Tree model saved to:")
print(model_path)


# ============================================================
# 8. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("DECISION TREE REGRESSION COMPLETED")
print("=" * 60)