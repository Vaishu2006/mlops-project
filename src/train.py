import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ✅ Load .env for local dev
load_dotenv()

# ✅ Best Fix: env variable → SQLite fallback (never file store)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Configurable via env
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))

print("Loading dataset...")

df = pd.read_csv("data/dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# ✅ Auto-create models/ directory if missing
os.makedirs("models", exist_ok=True)

mlflow.set_experiment("MLOpsDemo")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=42        # ✅ Reproducible results
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    # ✅ Log all params
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)

    # ✅ Log model to MLflow registry
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="MLOpsDemoModel"  # ✅ Auto-registers in Model Registry
    )

    # ✅ Also save locally as .pkl
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)  # ✅ Also track .pkl in MLflow

    print(f"✅ Accuracy  : {accuracy:.4f}")
    print(f"✅ Tracking  : {MLFLOW_TRACKING_URI}")
    print(f"✅ Model saved: {model_path}")