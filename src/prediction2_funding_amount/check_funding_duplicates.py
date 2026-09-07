import pandas as pd

print("=" * 60)
print("FUNDING AMOUNT - DUPLICATE CHECK")
print("=" * 60)

file_path = "datasets/funding_amount_prepared.csv"

df = pd.read_csv(file_path)

print("\nDataset Shape:")
print(df.shape)

# Check exact duplicate rows
exact_duplicates = df.duplicated().sum()

print("\nExact Duplicate Rows:")
print(exact_duplicates)

# Check duplicates based on startup-related features
startup_features = [
    "category_main",
    "market",
    "country_code",
    "region",
    "founded_year",
    "funding_rounds"
]

feature_duplicates = df.duplicated(
    subset=startup_features,
    keep=False
).sum()

print("\nRows with Duplicate Feature Combinations:")
print(feature_duplicates)

# Check duplicate feature combinations with different funding amounts
grouped = (
    df.groupby(startup_features)["funding_total_usd"]
    .nunique()
)

different_target_groups = (grouped > 1).sum()

print("\nFeature Groups with Different Funding Amounts:")
print(different_target_groups)

print("\n" + "=" * 60)
print("DUPLICATE CHECK COMPLETED")
print("=" * 60)