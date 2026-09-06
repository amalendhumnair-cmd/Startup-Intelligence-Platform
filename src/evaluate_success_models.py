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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_sample_weight


# --------------------------------------------------
# 1. LOAD DATASET
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
# 3. SELECT VALID STATUS VALUES
# --------------------------------------------------

df = df[
    df["status"].isin(["operating", "acquired", "closed"])
].copy()


# Success = 1
# Not Success = 0

df["success"] = df["status"].map({
    "operating": 1,
    "acquired": 1,
    "closed": 0
})


# --------------------------------------------------
# 4. CONVERT CATEGORY LIST
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
# 5. FEATURES AND TARGET
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
# 6. NUMERICAL AND CATEGORICAL FEATURES
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
# 7. PREPROCESSING
# --------------------------------------------------

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
# 8. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 9. CREATE MODELS
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

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}


# --------------------------------------------------
# 10. TRAIN AND EVALUATE
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # XGBoost gets balanced sample weights
    if model_name == "XGBoost":

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

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability for ROC-AUC
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------

    print("\nOverall Metrics:")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, y_pred), 4)
    )

    print(
        "Balanced Accuracy:",
        round(
            balanced_accuracy_score(y_test, y_pred),
            4
        )
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(y_test, y_probability),
            4
        )
    )


    # --------------------------------------------------
    # SUCCESS CLASS = 1
    # --------------------------------------------------

    print("\nSUCCESS (Class 1):")

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                y_pred,
                pos_label=1
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                y_pred,
                pos_label=1
            ),
            4
        )
    )

    print(
        "F1:",
        round(
            f1_score(
                y_test,
                y_pred,
                pos_label=1
            ),
            4
        )
    )


    # --------------------------------------------------
    # NOT SUCCESS CLASS = 0
    # --------------------------------------------------

    print("\nNOT SUCCESS (Class 0):")

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                y_pred,
                pos_label=0
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                y_pred,
                pos_label=0
            ),
            4
        )
    )

    print(
        "F1:",
        round(
            f1_score(
                y_test,
                y_pred,
                pos_label=0
            ),
            4
        )
    )


    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )