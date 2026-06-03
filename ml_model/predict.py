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
    predictor = StockPredictor()
    print(f"Model loaded: {predictor.model is not None}")
