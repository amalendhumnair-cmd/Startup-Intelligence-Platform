import pandas as pd

# ============================================================
# 1. LOAD CRUNCHBASE DATASET
# ============================================================

file_path = "datasets/investments_VC.csv"

df = pd.read_csv(
    file_path,
    encoding="latin1"
)

# Clean column names
df.columns = df.columns.str.strip()


# ============================================================
# 2. CLEAN FUNDING TOTAL
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
# 3. DISPLAY BASIC INFORMATION
# ============================================================

print("=" * 60)
print("FUNDING AMOUNT PREDICTION - DATASET INSPECTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nTarget Column:")
print("funding_total_usd")

print("\nTarget Data Type:")
print(df["funding_total_usd"].dtype)

print("\nMissing Target Values:")
print(df["funding_total_usd"].isnull().sum())

print("\nAvailable Target Values:")
print(df["funding_total_usd"].notnull().sum())


# ============================================================
# 4. TARGET STATISTICS
# ============================================================

print("\nTarget Statistics:")
print(df["funding_total_usd"].describe())


# ============================================================
# 5. CHECK ZERO VALUES
# ============================================================

print("\nZero Funding Values:")
print((df["funding_total_usd"] == 0).sum())


# ============================================================
# 6. CHECK NEGATIVE VALUES
# ============================================================

print("\nNegative Funding Values:")
print((df["funding_total_usd"] < 0).sum())


# ============================================================
# 7. DISPLAY ALL COLUMNS
# ============================================================

print("\nAll Dataset Columns:")

for column in df.columns:
    print("-", column)


# ============================================================
# 8. CHECK IMPORTANT FUNDING-RELATED COLUMNS
# ============================================================

funding_columns = [
    "funding_total_usd",
    "funding_rounds",
    "seed",
    "venture",
    "equity_crowdfunding",
    "undisclosed",
    "convertible_note",
    "debt_financing",
    "angel",
    "grant",
    "private_equity",
    "post_ipo_equity",
    "post_ipo_debt",
    "secondary_market",
    "product_crowdfunding",
    "round_A",
    "round_B",
    "round_C",
    "round_D",
    "round_E",
    "round_F",
    "round_G",
    "round_H"
]

print("\nFunding-Related Columns:")
print("=" * 60)

for column in funding_columns:

    if column in df.columns:

        print("\nColumn:", column)
        print("Data Type:", df[column].dtype)
        print("Missing:", df[column].isnull().sum())
        print("Non-Missing:", df[column].notnull().sum())


print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)