import pandas as pd
from sklearn.model_selection import train_test_split

# Load Dataset 1
file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

# Clean column names
df.columns = df.columns.str.strip()

# Clean funding amount
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

# Keep only rows with known status
df = df[df["status"].isin(["operating", "acquired", "closed"])].copy()

# Create target
df["success"] = df["status"].map({
    "operating": 1,
    "acquired": 1,
    "closed": 0
})

# Select input features
features = [
    "category_list",
    "market",
    "funding_total_usd",
    "funding_rounds",
    "country_code",
    "region",
    "founded_year"
]

# X = inputs
X = df[features]

# y = target
y = df["success"]

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

print("\nTotal data:", len(X))
print("Training data:", len(X_train))
print("Testing data:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nTraining target percentage:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target percentage:")
print(y_test.value_counts(normalize=True) * 100)