import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PREDICTION 2 - FUNDING AMOUNT
# TRAIN / TEST SPLIT
# ============================================================

# Prepared dataset path
file_path = "datasets/funding_amount_prepared.csv"

# Load prepared dataset
df = pd.read_csv(file_path)

print("=" * 60)
print("FUNDING AMOUNT - TRAIN / TEST SPLIT")
print("=" * 60)

print("\nPrepared Dataset Shape:")
print(df.shape)


# ============================================================
# 1. DEFINE FEATURES AND TARGET
# ============================================================

target = "funding_total_usd"

features = [
    "category_main",
    "market",
    "country_code",
    "region",
    "founded_year",
    "funding_rounds"
]

X = df[features]
y = df[target]


print("\nFeatures:")
for column in features:
    print("-", column)

print("\nTarget:")
print("-", target)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 3. DISPLAY SPLIT INFORMATION
# ============================================================

print("\nTraining Data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nTesting Data:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 4. SAVE TRAINING DATA
# ============================================================

train_data = X_train.copy()
train_data[target] = y_train

train_path = "datasets/funding_amount_train.csv"

train_data.to_csv(
    train_path,
    index=False
)


# ============================================================
# 5. SAVE TESTING DATA
# ============================================================

test_data = X_test.copy()
test_data[target] = y_test

test_path = "datasets/funding_amount_test.csv"

test_data.to_csv(
    test_path,
    index=False
)


# ============================================================
# 6. FINAL INFORMATION
# ============================================================

print("\nTraining dataset saved to:")
print(train_path)

print("\nTesting dataset saved to:")
print(test_path)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT COMPLETED")
print("=" * 60)