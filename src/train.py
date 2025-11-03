import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

os.makedirs("models", exist_ok=True)

train = pd.read_csv("data/train.csv")

X_train = train.drop("target", axis=1)
y_train = train["target"]

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

joblib.dump(model, "models/model.pkl")
