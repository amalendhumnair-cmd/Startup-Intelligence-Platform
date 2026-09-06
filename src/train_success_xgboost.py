import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

df.columns = df.columns.str.strip()


# --------------------------------------------------
# 2. Clean Funding Amount
# --------------------------------------------------

df["funding_total_usd"] = (
    df["funding_total_usd"]
    .astype(str)
    .str.strip()
    .str.replace(",", "", regex=False)
    .replace("-", pd.NA)
)

df["funding_total_usd"] = pd.to_numeric(
    df["funding_total_usd"],
    errors="coerce"
)


# --------------------------------------------------
# 3. Keep Known Status Values
# --------------------------------------------------

df = df[df["status"].isin([
    "operating",
    "acquired",
    "closed"
])].copy()


# --------------------------------------------------
# 4. Create Target
# --------------------------------------------------

df["success"] = df["status"].map({
    "operating": 1,
    "acquired": 1,
    "closed": 0
})


# --------------------------------------------------
# 5. Simplify Category
# --------------------------------------------------

def get_first_category(value):

    if pd.isna(value):
        return "Unknown"

    categories = [
        category.strip()
        for category in str(value).split("|")
        if category.strip()
    ]

    if len(categories) == 0:
        return "Unknown"

    return categories[0]


df["category_main"] = df["category_list"].apply(
    get_first_category
)


# --------------------------------------------------
# 6. Select Features
# --------------------------------------------------

features = [
    "category_main",
    "market",
    "funding_total_usd",
    "funding_rounds",
    "country_code",
    "region",
    "founded_year"
]

X = df[features]
y = df["success"]


# --------------------------------------------------
# 7. Feature Types
# --------------------------------------------------

numeric_features = [
    "funding_total_usd",
    "funding_rounds",
    "founded_year"
]

categorical_features = [
    "category_main",
    "market",
    "country_code",
    "region"
]


# --------------------------------------------------
# 8. Numerical Preprocessing
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# --------------------------------------------------
# 9. Categorical Preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# --------------------------------------------------
# 10. Combine Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 11. XGBoost Model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# --------------------------------------------------
# 12. Complete Pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


# --------------------------------------------------
# 13. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 14. Calculate Balanced Sample Weights
# --------------------------------------------------

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# --------------------------------------------------
# 15. Train Model
# --------------------------------------------------

print("=" * 60)
print("STARTUP SUCCESS PREDICTION")
print("XGBOOST")
print("=" * 60)

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weights
)

print("Training completed!")


# --------------------------------------------------
# 16. Make Predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 17. Calculate Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# --------------------------------------------------
# 18. Display Results
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=[
        "Not Success",
        "Success"
    ],
    zero_division=0
))


print("\nModel training and evaluation completed successfully!")