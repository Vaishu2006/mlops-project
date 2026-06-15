from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "MLOps API Running"}

@app.post("/predict")
def predict(feature1: int, feature2: int):

    prediction = model.predict(
        [[feature1, feature2]]
    )

    return {
        "prediction": int(prediction[0])
    }