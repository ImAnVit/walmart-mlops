import joblib
import numpy as np

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict(data):

    data_scaled = scaler.transform([data])

    prediction = model.predict(data_scaled)

    return float(prediction[0])