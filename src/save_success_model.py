import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

df.columns = df.columns.str.strip()


# ============================================================
# 2. CLEAN FUNDING COLUMN
# ============================================================

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


# ============================================================
# 3. KEEP VALID STATUS VALUES
# ============================================================

df = df[
    df["status"].isin(
        ["operating", "acquired", "closed"]
    )
].copy()


# ============================================================
# 4. CREATE TARGET
# ============================================================

df["success"] = df["status"].map({
    "operating": 1,
    "acquired": 1,
    "closed": 0
})


# ============================================================
# 5. PROCESS CATEGORY
# ============================================================

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


# ============================================================
# 6. SELECT FEATURES
# ============================================================

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


# ============================================================
# 7. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

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


# ============================================================
# 8. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ============================================================
# 9. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# ============================================================
# 10. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 12. XGBOOST BALANCED DEPTH MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# ============================================================
# 13. CREATE SAMPLE WEIGHTS
# ============================================================

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)


# ============================================================
# 14. CREATE COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])


# ============================================================
# 15. TRAIN MODEL
# ============================================================

pipeline.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weights
)


# ============================================================
# 16. SAVE MODEL + THRESHOLD
# ============================================================

selected_threshold = 0.45

model_data = {
    "pipeline": pipeline,
    "threshold": selected_threshold,
    "features": features
}

joblib.dump(
    model_data,
    "models/startup_success_xgboost.pkl"
)


# ============================================================
# 17. DISPLAY INFORMATION
# ============================================================

print("\n============================================================")
print("STARTUP SUCCESS MODEL SAVED")
print("============================================================")

print("\nModel: XGBoost Balanced Depth")

print("n_estimators: 200")
print("max_depth: 4")
print("learning_rate: 0.05")

print("\nSelected Threshold:", selected_threshold)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nSaved File:")
print("models/startup_success_xgboost.pkl")

print("\n============================================================")
print("MODEL SAVING COMPLETED")
print("============================================================")