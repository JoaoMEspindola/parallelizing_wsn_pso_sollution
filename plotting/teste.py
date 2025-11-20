import pandas as pd

df = pd.read_csv("gpu_block_network.csv")
print(df.columns)
print(df[df["recordType"]=="edge"].head(5))
