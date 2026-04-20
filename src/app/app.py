from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from contextlib import asynccontextmanager

# mlflow tracking URI
# mlflow model uri

tracking_uri = "http://localhost:5000"
model_uri = "mlflow-artifacts:/2/models/m-2ce0a3e4da824ca38d1217a6795a3809/artifacts"
 
model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(tracking_uri)
    model['pipeline'] = mlflow.sklearn.load_model(model_uri)
    yield
    model.clear()

app = FastAPI(
    title="Voyage ESG Forecast API",
    description="Forecast ESG scores using MLflow and scikit-learn models.",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    Average_Spend_GBP: float
    Total_Seats: int
    Seats_Sold_Realized: int
    Remaining_Seats_Realized: int
    Demand_Index: float
    Days_Before_Travel: int
    Price_Premium: float
    Load_Factor: float
    Seat_Class: Literal["Standard", "Flex", "First"]
    Booking_Channel: Literal["Web", "Mobile", "Phone", "Agent"]
    Origin: str
    Destination: str
    Route_Category: Literal["Long", "Medium", "Short"]
    Customer_Segment: Literal["Leisure", "Business", "Commuter", "Student"]
    Loyalty_Status: Literal["Gold", "Silver", "Bronze", "No_Loyalty"]

class PredictionResponse(BaseModel):
    ticket_price_prediction: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.model_dump()])
    prediction = model['pipeline'].predict(input_df)[0]

    print(input_df)
    return PredictionResponse(ticket_price_prediction=float(prediction))