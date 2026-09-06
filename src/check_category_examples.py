import pandas as pd

# Load Dataset 1
file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

# Clean column names
df.columns = df.columns.str.strip()

print("=" * 60)
print("CATEGORY LIST EXAMPLES")
print("=" * 60)

print("\nFirst 30 category values:\n")

for value in df["category_list"].dropna().head(30):
    print(value)