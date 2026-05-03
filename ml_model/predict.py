import mlflow.xgboost
import pandas as pd
import os

class StockPredictor:
    def __init__(self, model_name="stock_predictor", stage="Production"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        model_uri = f"models:/{model_name}/{stage}"
        try:
            self.model = mlflow.xgboost.load_model(model_uri)
            print(f"Loaded {model_name} from {stage} stage.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            
    def predict(self, features: pd.DataFrame):
        if not self.model:
            raise ValueError("Model is not loaded.")
        return self.model.predict(features)
        
if __name__ == "__main__":
    # Example usage
    predictor = StockPredictor()
