import pandas as pd

# Load the dataset
file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

# Display basic information
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatus Value Counts:")
print(df["status"].value_counts(dropna=False))