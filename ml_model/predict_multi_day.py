import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import os

MODEL_PATH = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/models/best_model.json"
PROCESSED_PATH = "/Volumes/workspace/default/stock_data/processed/stock_features.parquet"

def get_next_n_trading_dates(n=15):
    """
    Tính n ngày giao dịch tiếp theo (bỏ qua weekend)
    
    Returns:
        List of date strings ['2026-06-02', '2026-06-03', ...]
    """
    dates = []
    current = datetime.now() + timedelta(days=1)  # Start from tomorrow
    
    while len(dates) < n:
        # Skip weekend
        if current.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def predict_multi_day(n_days=15):
    """
    Dự đoán TẤT CẢ các mã cho N ngày giao dịch tiếp theo
    
    ⚠️  LƯU Ý:
    - Ngày 1 (day_1): Dự đoán chính xác nhất, dựa trên data mới nhất
    - Ngày 2-15: Dự đoán với giả định xu hướng tương tự
    - Confidence giảm dần theo thời gian (day_1 > day_2 > ... > day_15)
    
    Returns:
        predictions_df: DataFrame với columns:
            - ticker
            - latest_data_date
            - day_1_date, day_1_prediction, day_1_signal, day_1_probability, day_1_confidence
            - day_2_date, day_2_prediction, day_2_signal, day_2_probability, day_2_confidence
            - ...
            - day_15_date, day_15_prediction, ...
    """
    
    print("="*80)
    print(f"🔮 MULTI-DAY PREDICTION - DỰ ĐOÁN {n_days} NGÀY TIẾP THEO")
    print("="*80)
    
    # 1. Load model
    print(f"\n📦 Loading model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print("✅ Model loaded")
    
    # 2. Load latest data
    print(f"\n📂 Loading processed data...")
    df = pd.read_parquet(PROCESSED_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # Get latest data for each ticker
    latest_df = df.sort_values('date').groupby('ticker').tail(1).reset_index(drop=True)
    latest_date = latest_df['date'].max()
    
    print(f"   Latest data: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   Tickers: {len(latest_df)}")
    
    # 3. Extract base features
    print(f"\n🔧 Extracting features...")
    sample = latest_df['scaled_features'].iloc[0]
    if isinstance(sample, dict):
        X_base = np.vstack(latest_df['scaled_features'].apply(lambda x: x['values']).values)
    else:
        X_base = np.vstack(latest_df['scaled_features'].values)
    
    # 4. Get trading dates
    trading_dates = get_next_n_trading_dates(n_days)
    print(f"\n📅 Predicting for {n_days} trading days:")
    print(f"   From: {trading_dates[0]}")
    print(f"   To:   {trading_dates[-1]}")
    
    # 5. Multi-day predictions
    print(f"\n🤖 Making predictions...")
    
    result_data = {
        'ticker': latest_df['ticker'].values,
        'latest_data_date': latest_df['date'].dt.strftime('%Y-%m-%d').values
    }
    
    # Predict each day
    X_current = X_base.copy()
    
    for day_idx, pred_date in enumerate(trading_dates, 1):
        print(f"   Day {day_idx}/{n_days}: {pred_date}...", end=" ")
        
        # Predict
        predictions = model.predict(X_current)
        probabilities = model.predict_proba(X_current)[:, 1]
        
        # Adjust confidence based on day (decay factor)
        # Day 1: 100%, Day 2: 95%, ..., Day 15: 30%
        confidence_decay = max(0.3, 1.0 - (day_idx - 1) * 0.05)
        adjusted_probs = probabilities * confidence_decay + 0.5 * (1 - confidence_decay)
        
        # Calculate confidence levels
        def get_confidence(prob, day):
            # Stricter thresholds for later days
            base_threshold = 0.7 - (day - 1) * 0.02
            if prob >= base_threshold or prob <= (1 - base_threshold):
                return "HIGH"
            elif prob > 0.6 or prob < 0.4:
                return "MEDIUM"
            else:
                return "LOW"
        
        confidences = [get_confidence(p, day_idx) for p in adjusted_probs]
        
        # Store results
        result_data[f'day_{day_idx}_date'] = pred_date
        result_data[f'day_{day_idx}_prediction'] = predictions
        result_data[f'day_{day_idx}_signal'] = ['📈 TĂNG' if p == 1 else '📉 GIẢM' for p in predictions]
        result_data[f'day_{day_idx}_probability'] = adjusted_probs
        result_data[f'day_{day_idx}_confidence'] = confidences
        
        # Update features for next day (naive approach: slight trend continuation)
        # Adjust features slightly based on prediction
        for i in range(len(X_current)):
            if predictions[i] == 1:  # TĂNG
                X_current[i] *= 1.001  # Slight upward adjustment
            else:  # GIẢM
                X_current[i] *= 0.999  # Slight downward adjustment
        
        up_count = np.sum(predictions == 1)
        print(f"✅ {up_count}/{len(predictions)} TĂNG")
    
    # 6. Create DataFrame
    predictions_df = pd.DataFrame(result_data)
    
    # 7. Summary statistics
    print(f"\n📊 SUMMARY:")
    print("-" * 80)
    
    for day_idx in range(1, min(n_days + 1, 6)):  # Show first 5 days
        pred_col = f'day_{day_idx}_prediction'
        date_col = f'day_{day_idx}_date'
        
        up_count = np.sum(predictions_df[pred_col] == 1)
        down_count = np.sum(predictions_df[pred_col] == 0)
        total = len(predictions_df)
        
        date_val = predictions_df[date_col].iloc[0]
        print(f"   {date_val}: 📈 {up_count} TĂNG | 📉 {down_count} GIẢM ({up_count/total*100:.1f}% tăng)")
    
    if n_days > 5:
        print(f"   ... (và {n_days - 5} ngày nữa)")
    
    print("=" * 80)
    
    return predictions_df

if __name__ == "__main__":
    df = predict_multi_day(15)
    print(f"\n✅ Generated predictions for {len(df)} tickers × 15 days")
