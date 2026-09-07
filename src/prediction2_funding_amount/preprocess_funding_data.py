import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PREDICTION 2 - FUNDING AMOUNT
# DATA PREPROCESSING
# ============================================================

# File paths
train_path = "datasets/funding_amount_train.csv"
test_path = "datasets/funding_amount_test.csv"

# Load training and testing data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("=" * 60)
print("FUNDING AMOUNT - DATA PREPROCESSING")
print("=" * 60)

print("\nTraining Dataset Shape:")
print(train_df.shape)

print("\nTesting Dataset Shape:")
print(test_df.shape)


# ============================================================
# 1. DEFINE FEATURES AND TARGET
# ============================================================

features = [
    "category_main",
    "market",
    "country_code",
    "region",
    "founded_year",
    "funding_rounds"
]

target = "funding_total_usd"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# ============================================================
# 2. DEFINE NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "founded_year",
    "funding_rounds"
]

categorical_features = [
    "category_main",
    "market",
    "country_code",
    "region"
]


print("\nNumeric Features:")
for column in numeric_features:
    print("-", column)

print("\nCategorical Features:")
for column in categorical_features:
    print("-", column)


# ============================================================
# 3. NUMERIC PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# 4. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================================
# 5. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
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
    ]
)


# ============================================================
# 6. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 7. DISPLAY PREPROCESSED DATA INFORMATION
# ============================================================

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\nOriginal Training Missing Values:")
print(X_train.isnull().sum())

print("\nOriginal Testing Missing Values:")
print(X_test.isnull().sum())


# ============================================================
# 9. SAVE PREPROCESSOR
# ============================================================

preprocessor_path = "models/funding_amount_preprocessor.pkl"

joblib.dump(
    preprocessor,
    preprocessor_path
)

print("\nPreprocessor saved to:")
print(preprocessor_path)


# ============================================================
# 10. SAVE PROCESSED DATA
# ============================================================

joblib.dump(
    X_train_processed,
    "datasets/funding_amount_X_train.pkl"
)

joblib.dump(
    X_test_processed,
    "datasets/funding_amount_X_test.pkl"
)

joblib.dump(
    y_train,
    "datasets/funding_amount_y_train.pkl"
)

joblib.dump(
    y_test,
    "datasets/funding_amount_y_test.pkl"
)

print("\nProcessed training and testing data saved.")


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)