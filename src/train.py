import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Loading dataset...")

df = pd.read_csv("data/dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

mlflow.set_experiment("MLOpsDemo")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100
    )

    model.fit(X_train, y_train)

    accuracy = model.score(
        X_test, y_test
    )

    mlflow.log_param(
        "n_estimators", 100
    )

    mlflow.log_metric(
        "accuracy", accuracy
    )

    mlflow.sklearn.log_model(
        model, "model"
    )

    joblib.dump(
        model,
        "models/model.pkl"
    )

    print("Accuracy =", accuracy)
    print("Model saved")