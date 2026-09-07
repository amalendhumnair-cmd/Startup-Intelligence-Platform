import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


print("=" * 60)
print("PREDICTION 3 - SAVING FINAL XGBOOST MODEL")
print("=" * 60)


# --------------------------------------------------
# 1. Load prepared dataset
# --------------------------------------------------

file_path = "datasets/funding_rounds_prepared.csv"

df = pd.read_csv(file_path)

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
# 5. Preprocess training data
# --------------------------------------------------

X_train_processed = preprocessor.transform(
    X_train
)

print("\nProcessed training data:")
print(X_train_processed.shape)


# --------------------------------------------------
# 6. Create final XGBoost model
# --------------------------------------------------

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


# --------------------------------------------------
# 7. Train final model
# --------------------------------------------------

print("\nTraining final XGBoost model...")

model.fit(
    X_train_processed,
    y_train
)

print("Final model training completed.")


# --------------------------------------------------
# 8. Create final model package
# --------------------------------------------------

model_data = {
    "model": model,
    "preprocessor": preprocessor,
    "features": features,
    "target": target
}


# --------------------------------------------------
# 9. Save final model
# --------------------------------------------------

output_path = "models/funding_rounds_final.pkl"

joblib.dump(
    model_data,
    output_path
)

print("\nFinal model saved to:")
print(output_path)


print("\n" + "=" * 60)
print("FINAL MODEL SAVING COMPLETED")
print("=" * 60)