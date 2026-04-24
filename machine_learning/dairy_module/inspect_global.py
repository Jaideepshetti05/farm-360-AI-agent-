import pandas as pd

df = pd.read_csv("data/dairy/Milk Production By Country since 1960.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())
