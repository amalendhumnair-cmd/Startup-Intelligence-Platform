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

# Keep only known outcomes
df = df[df["status"].isin(["operating", "acquired", "closed"])].copy()

# Create target
df["success"] = df["status"].map({
    "operating": 1,
    "acquired": 1,
    "closed": 0
})

# Features we plan to use
features = [
    "category_list",
    "market",
    "funding_total_usd",
    "funding_rounds",
    "country_code",
    "region",
    "founded_year"
]

print("=" * 60)
print("LEAKAGE CHECK")
print("=" * 60)

print("\nTarget column:")
print("success")

print("\nInput features:")
for feature in features:
    print("-", feature)

print("\nColumns NOT being used:")
excluded = [column for column in df.columns if column not in features + ["success"]]
for column in excluded:
    print("-", column)

print("\nImportant:")
print("The target 'status' is NOT included as an input feature.")
print("Startup identifiers such as name and permalink are NOT used.")
print("Outcome-related columns are excluded from the current model.")

print("\nLeakage check completed.")