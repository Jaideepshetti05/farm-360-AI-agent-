import pandas as pd

# Load dataset
df = pd.read_csv("data/dairy/Mik_Pro.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nColumn Names:")
print(df.columns)
