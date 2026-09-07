import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 60)
print("FUNDING AMOUNT - LOCATION FEATURE DATA SPLIT")
print("=" * 60)

file_path = "datasets/funding_amount_location_prepared.csv"

df = pd.read_csv(file_path)

target = "funding_total_usd"

features = [
    "category_main",
    "market",
    "country_code",
    "state_code",
    "region",
    "city",
    "founded_year",
    "funding_rounds"
]

X = df[features]
y = df[target]

print("\nFull Dataset Shape:")
print(df.shape)

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Save training data
train_data = X_train.copy()
train_data[target] = y_train

train_data.to_csv(
    "datasets/funding_amount_location_train.csv",
    index=False
)

# Save testing data
test_data = X_test.copy()
test_data[target] = y_test

test_data.to_csv(
    "datasets/funding_amount_location_test.csv",
    index=False
)

print("\nTraining data saved to:")
print("datasets/funding_amount_location_train.csv")

print("\nTesting data saved to:")
print("datasets/funding_amount_location_test.csv")

print("\n" + "=" * 60)
print("LOCATION FEATURE DATA SPLIT COMPLETED")
print("=" * 60)