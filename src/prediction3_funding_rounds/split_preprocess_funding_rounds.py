import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


print("=" * 60)
print("PREDICTION 3 - SPLIT AND PREPROCESSING")
print("=" * 60)


# --------------------------------------------------
# 1. Load prepared dataset
# --------------------------------------------------

file_path = "datasets/funding_rounds_prepared.csv"

df = pd.read_csv(file_path)

print("\nPrepared dataset shape:")
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


print("\nFeatures:")
for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTrain/Test split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# --------------------------------------------------
# 4. Define column types
# --------------------------------------------------

categorical_features = [
    "category_main",
    "market",
    "country_code",
    "state_code",
    "region",
    "city"
]

numeric_features = [
    "funding_total_usd",
    "founded_year"
]


# --------------------------------------------------
# 5. Numeric preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# --------------------------------------------------
# 6. Categorical preprocessing
# --------------------------------------------------

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# --------------------------------------------------
# 7. Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------------------------
# 8. Fit preprocessing only on training data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nProcessed data shape:")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed :", X_test_processed.shape)


# --------------------------------------------------
# 9. Save preprocessor
# --------------------------------------------------

output_path = "models/funding_rounds_preprocessor.pkl"

joblib.dump(preprocessor, output_path)

print("\nPreprocessor saved to:")
print(output_path)


print("\n" + "=" * 60)
print("SPLIT AND PREPROCESSING COMPLETED")
print("=" * 60)