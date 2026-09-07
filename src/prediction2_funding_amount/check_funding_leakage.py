import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "datasets/investments_VC.csv"

df = pd.read_csv(
    file_path,
    encoding="latin1"
)

df.columns = df.columns.str.strip()


# ============================================================
# 2. CLEAN TARGET
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
# 3. FUNDING COMPONENT COLUMNS
# ============================================================

funding_columns = [
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
    "product_crowdfunding"
]


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


# ============================================================
# 4. CREATE SUM OF FUNDING COMPONENTS
# ============================================================

df["funding_components_sum"] = df[funding_columns].fillna(0).sum(axis=1)


# ============================================================
# 5. COMPARE COMPONENT SUM WITH TARGET
# ============================================================

comparison = df[
    df["funding_total_usd"].notna()
].copy()

comparison["difference"] = (
    comparison["funding_total_usd"]
    - comparison["funding_components_sum"]
)

comparison["absolute_difference"] = (
    comparison["difference"].abs()
)


# ============================================================
# 6. RESULTS
# ============================================================

print("=" * 60)
print("FUNDING AMOUNT - LEAKAGE CHECK")
print("=" * 60)

print("\nRows with valid funding total:")
print(len(comparison))

print("\nFunding component sum:")
print(comparison["funding_components_sum"].describe())

print("\nAbsolute difference between target and component sum:")
print(comparison["absolute_difference"].describe())


# ============================================================
# 7. EXACT MATCH CHECK
# ============================================================

exact_matches = (
    comparison["absolute_difference"] < 0.01
).sum()

print("\nExact / Near-Exact Matches:")
print(exact_matches)

print(
    "Percentage of matching rows:",
    round(exact_matches / len(comparison) * 100, 2),
    "%"
)


# ============================================================
# 8. CORRELATION WITH TARGET
# ============================================================

print("\nCorrelation with funding_total_usd:")
print("=" * 60)

for column in funding_columns + round_columns:

    correlation = df[
        ["funding_total_usd", column]
    ].corr().iloc[0, 1]

    print(
        f"{column:25s}: {correlation:.4f}"
    )


# ============================================================
# 9. ROUND COLUMN CHECK
# ============================================================

print("\nRound Columns:")
print("=" * 60)

for column in round_columns:

    non_zero = (
        df[column].fillna(0) > 0
    ).sum()

    print(
        f"{column:10s}: {non_zero} rows with funding"
    )


print("\n" + "=" * 60)
print("LEAKAGE CHECK COMPLETED")
print("=" * 60)