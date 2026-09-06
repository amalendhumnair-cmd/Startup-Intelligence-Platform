import pandas as pd

# Load Dataset 1
file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Clean funding values
df["funding_total_usd"] = (
    df["funding_total_usd"]
    .astype(str)
    .str.strip()
    .str.replace(",", "", regex=False)
    .replace("-", pd.NA)
)

# Convert funding amount to numeric
df["funding_total_usd"] = pd.to_numeric(
    df["funding_total_usd"],
    errors="coerce"
)

print("Dataset Shape:")
print(df.shape)

print("\nFunding Data Type:")
print(df["funding_total_usd"].dtype)

print("\nFirst 10 Cleaned Funding Values:")
print(df["funding_total_usd"].head(10))

print("\nMissing Funding Values:")
print(df["funding_total_usd"].isnull().sum())

print("\nStatus Values:")
print(df["status"].value_counts(dropna=False))