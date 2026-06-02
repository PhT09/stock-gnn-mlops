import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
import os

<<<<<<< Updated upstream
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
=======
MODEL_PATH = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/models/best_model.json"
PROCESSED_PATH = "/Volumes/workspace/default/stock_data/processed/stock_features.parquet"
PREDICTIONS_PATH = "/Volumes/workspace/default/stock_data/predictions/"

def get_next_trading_date():
    """Tính ngày giao dịch tiếp theo (bỏ qua weekend)"""
    today = datetime.now()
    next_date = today + timedelta(days=1)
>>>>>>> Stashed changes
    
    # Skip weekend
    while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_date += timedelta(days=1)
    
    return next_date.strftime('%Y-%m-%d')

def predict_next_session():
    """
    Dự đoán TẤT CẢ các mã cho phiên giao dịch tiếp theo
    
    Returns:
        predictions_df: DataFrame với columns:
            - prediction_date: Ngày đưa ra dự đoán (hôm nay)
            - ticker: Mã cổ phiếu
            - predicted_for_date: Ngày giao dịch được dự đoán (ngày mai/thứ 2)
            - prediction: 0 (giảm) hoặc 1 (tăng)
            - probability: Xác suất dự đoán (0-1)
            - confidence: HIGH/MEDIUM/LOW dựa vào probability
    """
    
    print("="*80)
    print("🔮 STOCK PREDICTION - DỰ ĐOÁN PHIÊN TIẾP THEO")
    print("="*80)
    
    # 1. Load best model
    print(f"\n📦 Loading best model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Please train model first!")
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
    
    # 2. Load processed data (latest data)
    print(f"\n📂 Loading processed data from: {PROCESSED_PATH}")
    df = pd.read_parquet(PROCESSED_PATH)
    print(f"   Total rows: {len(df):,}")
    print(f"   Tickers: {df['ticker'].nunique()}")
    
    # 3. Get latest data for each ticker (most recent date)
    print("\n🔍 Getting latest data for each ticker...")
    df['date'] = pd.to_datetime(df['date'])
    latest_df = df.sort_values('date').groupby('ticker').tail(1).reset_index(drop=True)
    
    latest_date = latest_df['date'].max()
    print(f"   Latest data date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   Tickers with latest data: {len(latest_df)}")
    
    # 4. Extract features
    print("\n🔧 Extracting features for prediction...")
    sample = latest_df['scaled_features'].iloc[0]
    if isinstance(sample, dict):
        # Old format
        X = np.vstack(latest_df['scaled_features'].apply(lambda x: x['values']).values)
    else:
        # New format
        X = np.vstack(latest_df['scaled_features'].values)
    
    # 5. Make predictions
    print("\n🤖 Making predictions for all tickers...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (tăng)
    
    # 6. Calculate confidence levels
    def get_confidence(prob):
        """
        Confidence level dựa vào probability:
        - HIGH: prob >= 0.7 or prob <= 0.3 (very confident)
        - MEDIUM: 0.3 < prob < 0.7 and (prob > 0.6 or prob < 0.4)
        - LOW: 0.4 <= prob <= 0.6 (uncertain)
        """
<<<<<<< Updated upstream
        aligned_features = self._align_features(features)
        return self.model.predict_proba(aligned_features)
        
=======
        if prob >= 0.7 or prob <= 0.3:
            return "HIGH"
        elif prob > 0.6 or prob < 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    confidences = [get_confidence(p) for p in probabilities]
    
    # 7. Create predictions dataframe
    prediction_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    predicted_for_date = get_next_trading_date()
    
    predictions_df = pd.DataFrame({
        'prediction_date': prediction_date,
        'ticker': latest_df['ticker'].values,
        'predicted_for_date': predicted_for_date,
        'prediction': predictions,
        'probability': probabilities,
        'confidence': confidences,
        'latest_data_date': latest_df['date'].dt.strftime('%Y-%m-%d').values
    })
    
    # 8. Add signal labels
    predictions_df['signal'] = predictions_df['prediction'].map({
        0: '📉 GIẢM',
        1: '📈 TĂNG'
    })
    
    # 9. Summary statistics
    print("\n📊 PREDICTION SUMMARY:")
    print("-" * 80)
    print(f"   Prediction date: {prediction_date}")
    print(f"   Predicted for date: {predicted_for_date} (phiên giao dịch tiếp theo)")
    print(f"   Total tickers: {len(predictions_df)}")
    print(f"\n   Signals:")
    print(f"      📈 TĂNG: {(predictions == 1).sum()} tickers ({(predictions == 1).sum()/len(predictions)*100:.1f}%)")
    print(f"      📉 GIẢM: {(predictions == 0).sum()} tickers ({(predictions == 0).sum()/len(predictions)*100:.1f}%)")
    print(f"\n   Confidence levels:")
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        count = (predictions_df['confidence'] == conf).sum()
        pct = count / len(predictions_df) * 100
        print(f"      {conf}: {count} tickers ({pct:.1f}%)")
    
    # 10. Top predictions
    print("\n🎯 TOP 10 DỰ ĐOÁN TĂNG MẠNH (High Confidence):")
    top_up = predictions_df[predictions_df['prediction'] == 1].sort_values('probability', ascending=False).head(10)
    for _, row in top_up.iterrows():
        print(f"   {row['ticker']:<8} 📈 {row['probability']:.1%} ({row['confidence']})")
    
    print("\n🎯 TOP 10 DỰ ĐOÁN GIẢM MẠNH (High Confidence):")
    top_down = predictions_df[predictions_df['prediction'] == 0].sort_values('probability').head(10)
    for _, row in top_down.iterrows():
        print(f"   {row['ticker']:<8} 📉 {1-row['probability']:.1%} ({row['confidence']})")
    
    # 11. Save to Unity Catalog
    print(f"\n💾 Saving predictions to: {PREDICTIONS_PATH}")
    spark = SparkSession.builder.appName("StockPredictions").getOrCreate()
    
    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(predictions_df)
    
    # Save as parquet (overwrite mode - mỗi lần chạy là predictions mới)
    spark_df.write.mode("overwrite").parquet(PREDICTIONS_PATH + "latest.parquet")
    
    # Also append to history table (keep all predictions history)
    try:
        spark_df.write.mode("append").parquet(PREDICTIONS_PATH + "history.parquet")
        print("   ✅ Saved to history table")
    except:
        # First time - create table
        spark_df.write.mode("overwrite").parquet(PREDICTIONS_PATH + "history.parquet")
        print("   ✅ Created history table")
    
    print("\n✅ PREDICTIONS COMPLETED!")
    print("="*80)
    print(f"\n📍 Query predictions:")
    print(f"   Latest: spark.read.parquet('{PREDICTIONS_PATH}latest.parquet')")
    print(f"   History: spark.read.parquet('{PREDICTIONS_PATH}history.parquet')")
    
    return predictions_df

>>>>>>> Stashed changes
if __name__ == "__main__":
    predictions = predict_next_session()
    print(f"\n✅ Generated {len(predictions)} predictions")
