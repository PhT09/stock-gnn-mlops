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
    
    def _align_features(self, features: pd.DataFrame):
        if not self.model:
            raise ValueError("Model is not loaded.")
        
        aligned = features.copy()
        try:
            booster = self.model.get_booster()
            expected_features = booster.feature_names
            if expected_features:
                for col in expected_features:
                    if col not in aligned.columns:
                        aligned[col] = 0.0
                aligned = aligned[expected_features]
            elif hasattr(self.model, "n_features_in_"):
                n_feats = self.model.n_features_in_
                if aligned.shape[1] != n_feats:
                    print(f"⚠️ Feature count mismatch. Model expects {n_feats}, input has {aligned.shape[1]}. Slicing/padding.")
                    if aligned.shape[1] > n_feats:
                        aligned = aligned.iloc[:, :n_feats]
                    else:
                        for i in range(aligned.shape[1], n_feats):
                            aligned[f"feat_{i}"] = 0.0
        except Exception as e:
            print(f"⚠️ Warning aligning features: {e}")
        return aligned

    def predict(self, features: pd.DataFrame):
        """
        Predict stock movement
        
        Args:
            features: DataFrame with 17 features [return_1d, return_3d, return_5d, 
                     return_10d, price_vs_ma5, price_vs_ma10, ma5_vs_ma10, 
                     volume_ratio, volume_change, volatility_5, volatility_10, 
                     oc_return, hl_range, close_position, return_lag1, 
                     return_lag2, return_lag3]
        
        Returns:
            predictions: 0=DOWN, 1=UP
        """
        aligned_features = self._align_features(features)
        return self.model.predict(aligned_features)
    
    def predict_proba(self, features: pd.DataFrame):
        """
        Predict probabilities
        
        Returns:
            probabilities: [prob_down, prob_up]
        """
        aligned_features = self._align_features(features)
        return self.model.predict_proba(aligned_features)


if __name__ == "__main__":
    # Test
    predictor = StockPredictorLocal()
    
    if predictor.model:
        print("\n🧪 Testing with dummy data...")
        features = pd.DataFrame({
            "return_1d": [0.01],
            "return_3d": [0.02],
            "return_5d": [0.03],
            "return_10d": [0.05],
            "price_vs_ma5": [1.02],
            "price_vs_ma10": [1.04],
            "ma5_vs_ma10": [1.02],
            "volume_ratio": [1.2],
            "volume_change": [0.1],
            "volatility_5": [0.015],
            "volatility_10": [0.018],
            "oc_return": [0.008],
            "hl_range": [0.02],
            "close_position": [0.75],
            "return_lag1": [0.005],
            "return_lag2": [-0.002],
            "return_lag3": [0.001]
        })
        
        pred = predictor.predict(features)
        proba = predictor.predict_proba(features)
        
        print(f"Prediction: {pred[0]} ({'UP' if pred[0] == 1 else 'DOWN'})")
        print(f"Confidence: DOWN={proba[0][0]:.2%}, UP={proba[0][1]:.2%}")
