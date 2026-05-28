import xgboost as xgb
import pandas as pd
import numpy as np

class StockPredictorLocal:
    """
    Load model từ file local (không cần Databricks token)
    Dùng cho BE khi không thể kết nối MLflow Registry
    """
    
    def __init__(self, model_path="models/best_model.json"):
        """
        Args:
            model_path: Đường dẫn tới file best_model.json
        """
        self.model = xgb.XGBClassifier()
        try:
            self.model.load_model(model_path)
            print(f"✅ Loaded model from: {model_path}")
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
    predictor = StockPredictorLocal()
    
    if predictor.model:
        print("\n🧪 Testing with dummy data...")
        features = pd.DataFrame({
            'ma5': [100.5],
            'ma10': [102.3],
            'ma20': [105.1],
            'rsi': [65.2],
            'macd': [1.2],
            'volatility_20': [0.02],
            'log_return': [0.01],
            'open': [100.0],
            'close': [101.0]
        })
        
        pred = predictor.predict(features)
        proba = predictor.predict_proba(features)
        
        print(f"Prediction: {pred[0]} ({'UP' if pred[0] == 1 else 'DOWN'})")
        print(f"Confidence: DOWN={proba[0][0]:.2%}, UP={proba[0][1]:.2%}")
