import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import json
import os

os.makedirs("results", exist_ok=True)

test = pd.read_csv("data/test.csv")
model = joblib.load("models/model.pkl")

X_test = test.drop("target", axis=1)
y_test = test["target"]

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

metrics = {"accuracy": acc}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f)
