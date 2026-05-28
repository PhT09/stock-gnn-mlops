import mlflow.xgboost
import pandas as pd
import os

class StockPredictor:
    def __init__(self, model_name="workspace.default.stock_predictor", alias="production"):
        """
        Load model từ MLflow Registry (Unity Catalog)
        
        Args:
            model_name: Full model name (workspace.default.stock_predictor)
            alias: Model alias (production, staging, champion...)
        """
        # Set tracking URI to Databricks
        mlflow.set_tracking_uri("databricks")
        
        # Unity Catalog models dùng alias thay vì stage
        model_uri = f"models:/{model_name}@{alias}"
        
        try:
            self.model = mlflow.xgboost.load_model(model_uri)
            print(f"✅ Loaded model: {model_name}@{alias}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model = None
            
    def predict(self, features: pd.DataFrame):
        """
        Predict stock movement
        
        Args:
            features: DataFrame with 9 features [ma5, ma10, ma20, rsi, macd, 
                     volatility_20, log_return, open, close]
        
        Returns:
            predictions: 0=DOWN, 1=UP
        """
        if not self.model:
            raise ValueError("Model is not loaded.")
        return self.model.predict(features)
    
    def predict_proba(self, features: pd.DataFrame):
        """
        Predict probabilities
        
        Returns:
            probabilities: [prob_down, prob_up]
        """
        if not self.model:
            raise ValueError("Model is not loaded.")
        return self.model.predict_proba(features)
        
if __name__ == "__main__":
    # Test
    predictor = StockPredictor()
    print(f"Model loaded: {predictor.model is not None}")
