import pandas as pd

df = pd.read_csv("nab_result.csv")

res = df.groupby("dataset")[['f1', 'recall', 'precission']].mean()

print(res)

df = pd.read_csv("yahoo_result.csv")

res = df.groupby("dataset")[['f1', 'recall', 'precission']].mean()

print(res)