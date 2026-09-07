import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("=" * 60)
print("FUNDING AMOUNT - LOCATION FEATURE PREPROCESSING")
print("=" * 60)

# Load training and testing data
train_path = "datasets/funding_amount_location_train.csv"
test_path = "datasets/funding_amount_location_test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("\nTraining Data Shape:")
print(train_df.shape)

print("\nTesting Data Shape:")
print(test_df.shape)

# Features and target
features = [
    "category_main",
    "market",
    "country_code",
    "state_code",
    "region",
    "city",
    "founded_year",
    "funding_rounds"
]

target = "funding_total_usd"

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# Separate numerical and categorical features
numeric_features = [
    "founded_year",
    "funding_rounds"
]

categorical_features = [
    "category_main",
    "market",
    "country_code",
    "state_code",
    "region",
    "city"
]

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nApplying preprocessing...")

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the same fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing completed.")

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)

# Save preprocessor
joblib.dump(
    preprocessor,
    "models/funding_amount_location_preprocessor.pkl"
)

# Save processed datasets
joblib.dump(
    X_train_processed,
    "datasets/funding_amount_location_X_train.pkl"
)

joblib.dump(
    X_test_processed,
    "datasets/funding_amount_location_X_test.pkl"
)

joblib.dump(
    y_train,
    "datasets/funding_amount_location_y_train.pkl"
)

joblib.dump(
    y_test,
    "datasets/funding_amount_location_y_test.pkl"
)

print("\nPreprocessor saved to:")
print("models/funding_amount_location_preprocessor.pkl")

print("\nProcessed data saved successfully.")

print("\n" + "=" * 60)
print("LOCATION FEATURE PREPROCESSING COMPLETED")
print("=" * 60)