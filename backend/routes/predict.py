from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from ml_model.predict import StockPredictor

router = APIRouter(prefix="/predict", tags=["predict"])

# We initialize the predictor globally so it's loaded once
predictor = None

class PredictionRequest(BaseModel):
    ticker: str
    features: dict

@router.post("/")
def predict(request: PredictionRequest):
    global predictor
    if predictor is None:
        try:
            predictor = StockPredictor()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")
            
    df = pd.DataFrame([request.features])
    try:
        prediction = predictor.predict(df)
        return {"ticker": request.ticker, "prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
