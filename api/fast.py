# write some code for the API here
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pandas as pd


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}


@app.get("/predict_fare")
def predict_fare(pickup_datetime,
                 pickup_longitude,
                 pickup_latitude,
                 dropoff_longitude,
                 dropoff_latitude,
                 passenger_count):

    X_pred = dict(key=['1'],
                  pickup_datetime=pickup_datetime,
                  pickup_longitude=pickup_longitude,
                  pickup_latitude=pickup_latitude,
                  dropoff_longitude=dropoff_longitude,
                  dropoff_latitude=dropoff_latitude,
                  passenger_count=passenger_count)
    X_pred = pd.DataFrame(X_pred)
    model = load('model.joblib')
    y_pred = model.predict(X_pred)

    return {"prediction": round(float(y_pred[0]), 2)}
