import pandas as pd

# Load Dataset 1
file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

# Clean column names
df.columns = df.columns.str.strip()

print("=" * 60)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 60)

categorical_features = [
    "category_list",
    "market",
    "country_code",
    "region"
]

for feature in categorical_features:
    print("\n" + feature)
    print("-" * 40)
    print("Unique values:", df[feature].nunique(dropna=True))
    print("Missing values:", df[feature].isna().sum())