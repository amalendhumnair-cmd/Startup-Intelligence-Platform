import pandas as pd

print("=" * 60)
print("PREDICTION 3 - FUNDING ROUNDS ANALYSIS")
print("=" * 60)

# Load dataset
file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

# Clean column names
df.columns = df.columns.str.strip()

print("\nDataset Shape:")
print(df.shape)

print("\nTarget Column:")
print("funding_rounds")

# ------------------------------------------------------------
# 1. Target information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET ANALYSIS")
print("=" * 60)

target = pd.to_numeric(df["funding_rounds"], errors="coerce")

print("\nMissing target values:")
print(target.isna().sum())

print("\nValid target values:")
print(target.notna().sum())

print("\nTarget statistics:")
print(target.describe())

print("\nUnique funding round values:")
print(sorted(target.dropna().unique()))

# ------------------------------------------------------------
# 2. Check suspicious columns
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("POSSIBLE LEAKAGE CHECK")
print("=" * 60)

possible_columns = [
    "funding_total_usd",
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

print("\nAvailable suspicious columns:")

for column in possible_columns:
    if column in df.columns:
        print("✓", column)

# ------------------------------------------------------------
# 3. Round column analysis
# ------------------------------------------------------------

round_columns = [
    "round_A",
    "round_B",
    "round_C",
    "round_D",
    "round_E",
    "round_F",
    "round_G",
    "round_H"
]

print("\n" + "=" * 60)
print("ROUND COLUMN CHECK")
print("=" * 60)

for column in round_columns:
    if column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        print(
            f"{column}: "
            f"non-zero = {(values.fillna(0) != 0).sum()}, "
            f"sum = {values.sum():,.2f}"
        )

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)