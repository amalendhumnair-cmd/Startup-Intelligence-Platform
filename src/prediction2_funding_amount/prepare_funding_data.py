import pandas as pd

# ============================================================
# PREDICTION 2 - FUNDING AMOUNT
# DATA PREPARATION
# ============================================================

# Dataset path
file_path = "datasets/investments_VC.csv"

# Load dataset
df = pd.read_csv(file_path, encoding="latin1")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("=" * 60)
print("FUNDING AMOUNT - DATA PREPARATION")
print("=" * 60)

print("\nOriginal Dataset Shape:")
print(df.shape)


# ============================================================
# 1. CLEAN TARGET COLUMN
# ============================================================

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


# ============================================================
# 2. KEEP ONLY ROWS WITH VALID TARGET
# ============================================================

df = df[df["funding_total_usd"].notna()].copy()

print("\nRows after removing missing funding amount:")
print(len(df))


# ============================================================
# 3. CONVERT FOUNDED YEAR TO NUMERIC
# ============================================================

df["founded_year"] = pd.to_numeric(
    df["founded_year"],
    errors="coerce"
)


# ============================================================
# 4. SELECT FEATURES
# ============================================================

features = [
    "category_list",
    "market",
    "country_code",
    "region",
    "founded_year",
    "funding_rounds"
]

target = "funding_total_usd"


# ============================================================
# 5. CREATE WORKING DATASET
# ============================================================

data = df[features + [target]].copy()


# ============================================================
# 6. CLEAN CATEGORY LIST
# ============================================================

data["category_list"] = (
    data["category_list"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

# Use the first category from category_list
data["category_main"] = (
    data["category_list"]
    .str.split("|")
    .str[0]
    .str.strip()
)

data["category_main"] = data["category_main"].replace(
    "",
    "Unknown"
)


# ============================================================
# 7. CLEAN CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "category_main",
    "market",
    "country_code",
    "region"
]

for column in categorical_columns:
    data[column] = (
        data[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ============================================================
# 8. CLEAN NUMERIC FEATURES
# ============================================================

data["funding_rounds"] = pd.to_numeric(
    data["funding_rounds"],
    errors="coerce"
)

data["founded_year"] = pd.to_numeric(
    data["founded_year"],
    errors="coerce"
)


# ============================================================
# 9. REMOVE ORIGINAL CATEGORY_LIST
# ============================================================

data = data.drop(columns=["category_list"])


# ============================================================
# 10. REMOVE INVALID TARGET VALUES
# ============================================================

data = data[
    data["funding_total_usd"] > 0
].copy()


# ============================================================
# 11. DISPLAY FINAL DATASET
# ============================================================

print("\nFinal Features:")
for column in data.drop(columns=[target]).columns:
    print("-", column)

print("\nTarget:")
print("-", target)

print("\nFinal Dataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nFinal Data Preview:")
print(data.head())


# ============================================================
# 12. SAVE PREPARED DATASET
# ============================================================

output_path = "datasets/funding_amount_prepared.csv"

data.to_csv(
    output_path,
    index=False
)

print("\nPrepared dataset saved to:")
print(output_path)

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETED")
print("=" * 60)