from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "Bank Note Authentication API Running"}

@app.post("/predict")
def predict(
    variance: float,
    skewness: float,
    curtosis: float,
    entropy: float
):

    data = pd.DataFrame(
        [[variance, skewness, curtosis, entropy]],
        columns=["variance", "skewness", "curtosis", "entropy"]
    )

    prediction = model.predict(data)

    result = "Fake Note" if prediction[0] == 1 else "Genuine Note"

    return {
        "prediction": int(prediction[0]),
        "result": result
    }