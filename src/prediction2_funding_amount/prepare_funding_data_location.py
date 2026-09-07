import pandas as pd

print("=" * 60)
print("FUNDING AMOUNT - LOCATION FEATURE EXPERIMENT")
print("=" * 60)

file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")
df.columns = df.columns.str.strip()

print("\nOriginal Dataset Shape:")
print(df.shape)

# Clean target
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

# Keep rows with valid target
df = df[df["funding_total_usd"].notna()].copy()

# Clean numeric features
df["funding_rounds"] = pd.to_numeric(
    df["funding_rounds"],
    errors="coerce"
)

df["founded_year"] = pd.to_numeric(
    df["founded_year"],
    errors="coerce"
)

# Create main category
df["category_list"] = (
    df["category_list"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

df["category_main"] = (
    df["category_list"]
    .str.split("|")
    .str[0]
    .str.strip()
)

df["category_main"] = df["category_main"].replace(
    "",
    "Unknown"
)

# Features including additional location information
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

data = df[features + [target]].copy()

# Clean categorical columns
categorical_columns = [
    "category_main",
    "market",
    "country_code",
    "state_code",
    "region",
    "city"
]

for column in categorical_columns:
    data[column] = (
        data[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

# Keep only positive funding amounts
data = data[data[target] > 0].copy()

print("\nFinal Features:")
for column in features:
    print("-", column)

print("\nTarget:")
print("-", target)

print("\nFinal Dataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

output_path = "datasets/funding_amount_location_prepared.csv"

data.to_csv(
    output_path,
    index=False
)

print("\nPrepared dataset saved to:")
print(output_path)

print("\n" + "=" * 60)
print("LOCATION FEATURE PREPARATION COMPLETED")
print("=" * 60)