from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to import ml_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model.inference import StockPredictor
import pandas as pd
import json

app = FastAPI(title="Stock GNN MLOps API", version="1.0.0")

# Define base directory (root of the project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path helpers
def get_path(rel_path):
    return os.path.join(BASE_DIR, rel_path.lstrip("./").lstrip("../"))

# Initialize Predictor
try:
    predictor = StockPredictor(
        model_path=get_path("ml_model/best_model.pt"),
        features_path=get_path("data/processed/mock_stocks.parquet"),
        edges_path=get_path("data/graph/mock_edges.csv"),
        embeddings_path=get_path("data/graph/mock_embeddings.npy")
    )
except Exception as e:
    print(f"Warning: StockPredictor failed to initialize: {e}")
    predictor = None

class PredictionRequest(BaseModel):
    ticker: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Stock GNN MLOps API", "status": "running"}

@app.get("/data")
def get_data():
    """Returns the latest processed mock data as JSON."""
    try:
        path = get_path("data/processed/mock_stocks.parquet")
        df = pd.read_parquet(path)
        return df.tail(20).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Path checked: {path}. Error: {str(e)}")

@app.get("/graph")
def get_graph():
    """Returns graph edges for visualization."""
    try:
        path = get_path("data/graph/mock_edges.csv")
        edge_df = pd.read_csv(path)
        return edge_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Path checked: {path}. Error: {str(e)}")

@app.post("/predict")
def predict_trend(request: PredictionRequest):
    """Predicts trend for a given ticker."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model service is currently unavailable.")
    
    try:
        result = predictor.predict(request.ticker)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
