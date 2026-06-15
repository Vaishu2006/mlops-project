import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load .env for local dev
load_dotenv()

# MLflow setup — artifact root as absolute path to avoid Windows path issues on Linux CI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
ARTIFACT_ROOT = os.path.abspath("./mlruns")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create experiment with explicit artifact location if it doesn't exist
exp = mlflow.get_experiment_by_name("MLOpsDemo")
if exp is None:
    mlflow.create_experiment("MLOpsDemo", artifact_location=f"file://{ARTIFACT_ROOT}")
mlflow.set_experiment("MLOpsDemo")

# Configurable via env
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))

print("Loading dataset...")

df = pd.read_csv("data/BankNoteAuthentication.csv")

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Auto-create models/ directory if missing
os.makedirs("models", exist_ok=True)

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    # Log params
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)

    # Log model to MLflow (no registered_model_name — requires remote server)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    # Save locally as .pkl and track in MLflow
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    print(f"✅ Accuracy  : {accuracy:.4f}")
    print(f"✅ Tracking  : {MLFLOW_TRACKING_URI}")
    print(f"✅ Model saved: {model_path}")