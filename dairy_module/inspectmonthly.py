import pandas as pd

df = pd.read_csv("data/dairy/monthly-milk-production-pounds.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())
