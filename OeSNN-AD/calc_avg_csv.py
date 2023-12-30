"""
    Script that calc average results for all datasets.
"""

import pandas as pd

print("---------NAB---------")

df = pd.read_csv("results/nab_result.csv")
res = df.groupby("dataset")[['f1', 'recall', 'precission']].mean()
print("BEZ:")
print(res)

df = pd.read_csv("results/ceemdan_nab_result-committee.csv")
res = df.groupby("dataset")[['f1', 'recall', 'precission']].mean()
print("CEEMDAN (committeee):")
print(res)

df = pd.read_csv("results/ceemdan_nab_result-geometric.csv")
res = df.groupby("dataset")[['f1', 'recall', 'precission']].mean()
print("CEEMDAN (geometric):")
print(res)

print("---------YAHOO---------")

df = pd.read_csv("results/yahoo_result.csv")
res = df.groupby("dataset")[['f1', 'recall', 'precission']].mean()
print("BEZ:")
print(res)

df = pd.read_csv("results/ceemdan_yahoo_result-committee.csv")
res = df.groupby("dataset")[['f1', 'recall', 'precission']].mean()
print("CEEMDAN (committeee):")
print(res)

df = pd.read_csv("results/ceemdan_yahoo_result-geometric.csv")
res = df.groupby("dataset")[['f1', 'recall', 'precission']].mean()
print("CEEMDAN (geometric):")
print(res)
