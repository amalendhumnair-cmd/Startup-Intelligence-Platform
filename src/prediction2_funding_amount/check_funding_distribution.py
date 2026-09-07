import pandas as pd
import numpy as np

print("=" * 60)
print("FUNDING AMOUNT - TARGET DISTRIBUTION CHECK")
print("=" * 60)

file_path = "datasets/funding_amount_prepared.csv"

df = pd.read_csv(file_path)

target = df["funding_total_usd"]

print("\nTarget Statistics:")
print("-" * 60)
print(f"Minimum : ${target.min():,.2f}")
print(f"25%     : ${target.quantile(0.25):,.2f}")
print(f"Median  : ${target.median():,.2f}")
print(f"75%     : ${target.quantile(0.75):,.2f}")
print(f"Maximum : ${target.max():,.2f}")
print(f"Mean    : ${target.mean():,.2f}")

print("\nSelected Percentiles:")
print("-" * 60)

for p in [0.90, 0.95, 0.99, 0.995]:
    value = target.quantile(p)
    print(f"{p * 100:.1f}% : ${value:,.2f}")

print("\nLargest 10 Funding Amounts:")
print("-" * 60)
print(target.nlargest(10).to_string(index=False))

print("\nLog-Transformed Target Statistics:")
print("-" * 60)

log_target = np.log1p(target)

print(f"Minimum : {log_target.min():.4f}")
print(f"Median  : {log_target.median():.4f}")
print(f"Maximum : {log_target.max():.4f}")
print(f"Mean    : {log_target.mean():.4f}")

print("\n" + "=" * 60)
print("DISTRIBUTION CHECK COMPLETED")
print("=" * 60)