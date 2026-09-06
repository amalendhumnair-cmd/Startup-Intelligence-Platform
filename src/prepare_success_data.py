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

# Keep only rows with a known startup status
df = df[df["status"].isin(["operating", "acquired", "closed"])].copy()

# Create Success Target
df["success"] = df["status"].map({
    "operating": 1,
    "acquired": 1,
    "closed": 0
})

print("Dataset Shape After Target Preparation:")
print(df.shape)

print("\nOriginal Status:")
print(df["status"].value_counts())

print("\nNew Success Target:")
print(df["success"].value_counts())

print("\nSuccess Target Meaning:")
print("1 = SUCCESS (Operating or Acquired)")
print("0 = NOT SUCCESS (Closed)")