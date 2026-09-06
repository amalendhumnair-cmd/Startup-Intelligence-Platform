import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
    f1_score
)

from sklearn.utils.class_weight import compute_sample_weight


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()


# --------------------------------------------------
# 2. CLEAN FUNDING COLUMN
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
# 3. CREATE TARGET
# --------------------------------------------------

df = df[
    df["status"].isin(["operating", "acquired", "closed"])
].copy()

df["success"] = df["status"].map({
    "operating": 1,
    "acquired": 1,
    "closed": 0
})


# --------------------------------------------------
# 4. CATEGORY PROCESSING
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
# 5. FEATURES
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
# 6. PREPROCESSING
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

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# --------------------------------------------------
# 7. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 8. MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost Basic": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ),

    "XGBoost Balanced Depth": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    ),

    "XGBoost Deeper": XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}


# --------------------------------------------------
# 9. STORE RESULTS
# --------------------------------------------------

results = []


# --------------------------------------------------
# 10. TRAIN AND EVALUATE
# --------------------------------------------------

for model_name, model in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    if model_name.startswith("XGBoost"):

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        pipeline.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights
        )

    else:

        pipeline.fit(
            X_train,
            y_train
        )

    y_pred = pipeline.predict(X_test)

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    results.append({
        "Model": model_name,

        "Accuracy": round(
            accuracy_score(y_test, y_pred),
            4
        ),

        "Balanced Accuracy": round(
            balanced_accuracy_score(
                y_test,
                y_pred
            ),
            4
        ),

        "ROC-AUC": round(
            roc_auc_score(
                y_test,
                y_probability
            ),
            4
        ),

        "Not Success Recall": round(
            recall_score(
                y_test,
                y_pred,
                pos_label=0
            ),
            4
        ),

        "Not Success F1": round(
            f1_score(
                y_test,
                y_pred,
                pos_label=0
            ),
            4
        )
    })


# --------------------------------------------------
# 11. CREATE COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 90)
print("FINAL MODEL COMPARISON")
print("=" * 90)

print(
    results_df.to_string(index=False)
)