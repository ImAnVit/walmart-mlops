from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")


class SalesRequest(BaseModel):
    Store: int
    Holiday_Flag: int
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float
    year: int
    month: int
    week: int


@app.get("/")
def home():
    return {"message": "Walmart Sales Prediction API"}


@app.post("/predict")
def predict_sales(data: SalesRequest):

    features = np.array([[

        data.Store,
        data.Holiday_Flag,
        data.Temperature,
        data.Fuel_Price,
        data.CPI,
        data.Unemployment,
        data.year,
        data.month,
        data.week

    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    return {"predicted_sales": float(prediction[0])}