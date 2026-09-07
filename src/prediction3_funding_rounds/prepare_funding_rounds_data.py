import pandas as pd

print("=" * 60)
print("PREDICTION 3 - PREPARING FUNDING ROUNDS DATA")
print("=" * 60)

# Load dataset
file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

# Clean column names
df.columns = df.columns.str.strip()

print("\nOriginal dataset shape:")
print(df.shape)

# ------------------------------------------------------------
# 1. Clean target
# ------------------------------------------------------------

df["funding_rounds"] = pd.to_numeric(
    df["funding_rounds"],
    errors="coerce"
)

# Keep only rows with valid target
df = df.dropna(subset=["funding_rounds"]).copy()

print("\nRows with valid funding_rounds:")
print(len(df))

# ------------------------------------------------------------
# 2. Create category_main
# ------------------------------------------------------------

def get_main_category(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip()

    if value == "":
        return "Unknown"

    return value.split("|")[0].strip()


df["category_main"] = df["category_list"].apply(get_main_category)

# ------------------------------------------------------------
# 3. Clean funding_total_usd
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 4. Select legitimate features
# ------------------------------------------------------------

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

final_columns = features + [target]

prepared_df = df[final_columns].copy()

# ------------------------------------------------------------
# 5. Clean founded_year
# ------------------------------------------------------------

prepared_df["founded_year"] = pd.to_numeric(
    prepared_df["founded_year"],
    errors="coerce"
)

# ------------------------------------------------------------
# 6. Remove invalid target values
# ------------------------------------------------------------

prepared_df = prepared_df[
    prepared_df["funding_rounds"] > 0
].copy()

# ------------------------------------------------------------
# 7. Display final information
# ------------------------------------------------------------

print("\nFinal dataset shape:")
print(prepared_df.shape)

print("\nFeatures:")
for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)

print("\nMissing values:")
print(prepared_df.isnull().sum())

print("\nFunding rounds distribution:")
print(prepared_df["funding_rounds"].value_counts().sort_index())

# ------------------------------------------------------------
# 8. Save prepared dataset
# ------------------------------------------------------------

output_path = "datasets/funding_rounds_prepared.csv"

prepared_df.to_csv(
    output_path,
    index=False
)

print("\nPrepared dataset saved to:")
print(output_path)

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETED")
print("=" * 60)