import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
    f1_score
)

from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()


# --------------------------------------------------
# 2. CLEAN FUNDING
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
# 3. TARGET
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
# 4. CATEGORY
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
# 8. THREE XGBOOST MODELS
# --------------------------------------------------

models = {

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
# 9. TRAIN AND COMPARE
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print(model_name)
    print("=" * 65)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    pipeline.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weights
    )

    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Use 0.50 threshold for this experiment
    y_pred = (y_probability >= 0.50).astype(int)

    print("\nAccuracy:")
    print(round(accuracy_score(y_test, y_pred), 4))

    print("\nBalanced Accuracy:")
    print(round(
        balanced_accuracy_score(y_test, y_pred),
        4
    ))

    print("\nROC-AUC:")
    print(round(
        roc_auc_score(y_test, y_probability),
        4
    ))

    print("\nNot Success Recall:")
    print(round(
        recall_score(
            y_test,
            y_pred,
            pos_label=0
        ),
        4
    ))

    print("\nNot Success F1:")
    print(round(
        f1_score(
            y_test,
            y_pred,
            pos_label=0
        ),
        4
    ))