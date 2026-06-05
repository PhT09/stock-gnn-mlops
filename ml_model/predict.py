import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
import os

MODEL_PATH = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/models/best_model.json"
PROCESSED_PATH = "/Volumes/workspace/default/stock_data/processed/stock_features.parquet"
PREDICTIONS_PATH = "/Volumes/workspace/default/stock_data/predictions/"

def get_next_trading_date():
    """Tính ngày giao dịch tiếp theo (bỏ qua weekend)"""
    today = datetime.now()
    next_date = today + timedelta(days=1)
    
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
    # Handle session markers in date (e.g., "2026-06-02(2)" -> "2026-06-02")
    df['date'] = df['date'].astype(str).str.replace(r'\(\d+\)$', '', regex=True)
    df['date'] = pd.to_datetime(df['date'])
    latest_df = df.sort_values('date').groupby('ticker').tail(1).reset_index(drop=True)
    
    latest_date = latest_df['date'].max()
    print(f"   Latest data date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   Tickers with latest data: {len(latest_df)}")
    
    # 4. Extract features - NEW FORMAT: individual scaled columns
    print("\n🔧 Extracting features for prediction...")
    
    feature_cols = [
        "return_1d_scaled",
        "return_3d_scaled",
        "return_5d_scaled",
        "return_10d_scaled",
        "price_vs_ma5_scaled",
        "price_vs_ma10_scaled",
        "ma5_vs_ma10_scaled",
        "volume_ratio_scaled",
        "volume_change_scaled",
        "volatility_5_scaled",
        "volatility_10_scaled",
        "oc_return_scaled",
        "hl_range_scaled",
        "close_position_scaled",
        "return_lag1_scaled",
        "return_lag2_scaled",
        "return_lag3_scaled"
    ]
    
    X = latest_df[feature_cols].values
    print(f"   Feature shape: {X.shape}")
    
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
        if prob >= 0.7 or prob <= 0.3:
            return "HIGH"
        elif prob > 0.6 or prob < 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    confidences = [get_confidence(p) for p in probabilities]
    
    # 7. Create predictions dataframe
    predicted_for = get_next_trading_date()
    
    predictions_df = pd.DataFrame({
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        'ticker': latest_df['ticker'].values,
        'latest_data_date': latest_df['date'].dt.strftime('%Y-%m-%d').values,
        'predicted_for_date': predicted_for,
        'prediction': predictions,
        'probability': probabilities,
        'confidence': confidences
    })
    
    # 8. Summary
    up_count = np.sum(predictions == 1)
    down_count = np.sum(predictions == 0)
    total = len(predictions)
    
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"   • Total tickers: {total}")
    print(f"   • Predicted UP (1): {up_count} ({up_count/total*100:.1f}%)")
    print(f"   • Predicted DOWN (0): {down_count} ({down_count/total*100:.1f}%)")
    print(f"   • Predicting for date: {predicted_for}")
    
    # Confidence distribution
    high_conf = sum([1 for c in confidences if c == "HIGH"])
    med_conf = sum([1 for c in confidences if c == "MEDIUM"])
    low_conf = sum([1 for c in confidences if c == "LOW"])
    
    print(f"\n🎯 CONFIDENCE DISTRIBUTION:")
    print(f"   • HIGH: {high_conf} ({high_conf/total*100:.1f}%)")
    print(f"   • MEDIUM: {med_conf} ({med_conf/total*100:.1f}%)")
    print(f"   • LOW: {low_conf} ({low_conf/total*100:.1f}%)")
    
    # 9. Save predictions
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    output_file = os.path.join(PREDICTIONS_PATH, f"predictions_{predicted_for}.parquet")
    predictions_df.to_parquet(output_file, index=False)
    print(f"\n💾 Predictions saved to: {output_file}")
    
    print("\n" + "="*80)
    print("✅ PREDICTION COMPLETED")
    print("="*80)
    
    return predictions_df

if __name__ == "__main__":
    predictions_df = predict_next_session()
    
    # Show top 10 high-confidence UP predictions
    print("\n🔥 TOP 10 HIGH-CONFIDENCE UP PREDICTIONS:")
    top_up = predictions_df[
        (predictions_df['prediction'] == 1) & 
        (predictions_df['confidence'] == 'HIGH')
    ].nlargest(10, 'probability')
    
    if len(top_up) > 0:
        print(top_up[['ticker', 'probability', 'confidence']].to_string(index=False))
    else:
        print("   No high-confidence UP predictions")
