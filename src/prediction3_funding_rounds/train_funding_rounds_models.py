import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)
print("PREDICTION 3 - FUNDING ROUNDS MODEL COMPARISON")
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

from sklearn.model_selection import train_test_split

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
# 5. Transform the data
# --------------------------------------------------

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print("\nProcessed data:")
print("X_train:", X_train_processed.shape)
print("X_test :", X_test_processed.shape)


# --------------------------------------------------
# 6. Define models
# --------------------------------------------------

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 7. Train and evaluate models
# --------------------------------------------------

results = []


for name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", name)
    print("-" * 60)

    # Train
    model.fit(X_train_processed, y_train)

    # Predict
    y_pred = model.predict(X_test_processed)

    # Prevent negative predictions
    y_pred = np.maximum(y_pred, 0)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# --------------------------------------------------
# 8. Create results table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

results_path = "results/prediction3_funding_rounds_results.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:")
print(results_path)


# --------------------------------------------------
# 10. Select best model based on R²
# --------------------------------------------------

best_model_name = results_df.loc[
    results_df["R2"].idxmax(),
    "Model"
]

print("\nBest model based on R²:")
print(best_model_name)


print("\n" + "=" * 60)
print("MODEL COMPARISON COMPLETED")
print("=" * 60)