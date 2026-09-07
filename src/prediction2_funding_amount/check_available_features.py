import pandas as pd

print("=" * 60)
print("FUNDING AMOUNT - AVAILABLE FEATURE CHECK")
print("=" * 60)

file_path = "datasets/investments_VC.csv"

df = pd.read_csv(file_path, encoding="latin1")

df.columns = df.columns.str.strip()

print("\nTotal Columns:")
print(len(df.columns))

print("\nAll Available Columns:")
print("-" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\n" + "=" * 60)
print("FEATURE CHECK COMPLETED")
print("=" * 60)