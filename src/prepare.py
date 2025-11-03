import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/data.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
