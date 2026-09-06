import pandas as pd

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

# Keep rows with known status
df = df[df["status"].isin(["operating", "acquired", "closed"])].copy()

# Create target
df["success"] = df["status"].map({
    "operating": 1,
    "acquired": 1,
    "closed": 0
})

# Features for Startup Success Prediction
features = [
    "category_list",
    "market",
    "funding_total_usd",
    "funding_rounds",
    "country_code",
    "region",
    "founded_year"
]

print("SUCCESS PREDICTION FEATURES")
print("=" * 50)

print("\nSelected Features:")
for feature in features:
    print("-", feature)

print("\nMissing Values in Selected Features:")
print(df[features].isnull().sum())

print("\nData Types:")
print(df[features].dtypes)

print("\nTotal Rows:")
print(len(df))

print("\nTarget Distribution:")
print(df["success"].value_counts())