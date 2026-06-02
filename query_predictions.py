"""
Helper script để query predictions dễ dàng
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PREDICTIONS_PATH = "/Volumes/workspace/default/stock_data/predictions/"

def get_latest_predictions():
    """Lấy predictions mới nhất"""
    spark = SparkSession.builder.appName("QueryPredictions").getOrCreate()
    df = spark.read.parquet(PREDICTIONS_PATH + "latest.parquet")
    return df

def get_predictions_for_ticker(ticker):
    """Lấy prediction cho 1 mã cụ thể"""
    df = get_latest_predictions()
    return df.filter(F.col("ticker") == ticker)

def get_predictions_by_signal(signal):
    """
    Lấy predictions theo signal
    signal: 1 (tăng) hoặc 0 (giảm)
    """
    df = get_latest_predictions()
    return df.filter(F.col("prediction") == signal).orderBy(F.desc("probability"))

def get_high_confidence_predictions():
    """Lấy predictions HIGH confidence"""
    df = get_latest_predictions()
    return df.filter(F.col("confidence") == "HIGH").orderBy(F.desc("probability"))

def get_predictions_history(ticker=None, days=30):
    """
    Lấy lịch sử predictions
    ticker: Mã cụ thể (None = tất cả)
    days: Số ngày gần nhất
    """
    spark = SparkSession.builder.appName("QueryPredictions").getOrCreate()
    df = spark.read.parquet(PREDICTIONS_PATH + "history.parquet")
    
    # Filter by date
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    df = df.filter(F.col("prediction_date") >= cutoff_date)
    
    # Filter by ticker if specified
    if ticker:
        df = df.filter(F.col("ticker") == ticker)
    
    return df.orderBy(F.desc("prediction_date"))

# ============================================================
# QUICK ACCESS FUNCTIONS
# ============================================================

def show_latest_summary():
    """Hiển thị tóm tắt predictions mới nhất"""
    df = get_latest_predictions()
    
    print("="*80)
    print("📊 PREDICTIONS SUMMARY (LATEST)")
    print("="*80)
    
    # Basic info
    pdf = df.toPandas()
    print(f"\nPrediction Date: {pdf['prediction_date'].iloc[0]}")
    print(f"Predicted For: {pdf['predicted_for_date'].iloc[0]}")
    print(f"Total Tickers: {len(pdf)}")
    
    # Signals
    print(f"\n📈 TĂNG: {(pdf['prediction'] == 1).sum()} tickers ({(pdf['prediction'] == 1).sum()/len(pdf)*100:.1f}%)")
    print(f"📉 GIẢM: {(pdf['prediction'] == 0).sum()} tickers ({(pdf['prediction'] == 0).sum()/len(pdf)*100:.1f}%)")
    
    # Confidence
    print(f"\nConfidence Distribution:")
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        count = (pdf['confidence'] == conf).sum()
        print(f"   {conf}: {count} ({count/len(pdf)*100:.1f}%)")
    
    # Top predictions
    print(f"\n🎯 TOP 10 TĂNG (High Confidence):")
    top_up = pdf[pdf['prediction'] == 1].sort_values('probability', ascending=False).head(10)
    for _, row in top_up.iterrows():
        print(f"   {row['ticker']:<8} 📈 {row['probability']:.1%} ({row['confidence']})")
    
    print(f"\n🎯 TOP 10 GIẢM (High Confidence):")
    top_down = pdf[pdf['prediction'] == 0].sort_values('probability').head(10)
    for _, row in top_down.iterrows():
        print(f"   {row['ticker']:<8} 📉 {1-row['probability']:.1%} ({row['confidence']})")
    
    print("="*80)

def find_ticker(ticker):
    """Tìm prediction cho 1 mã cụ thể"""
    df = get_predictions_for_ticker(ticker)
    pdf = df.toPandas()
    
    if len(pdf) == 0:
        print(f"❌ Không tìm thấy prediction cho {ticker}")
        return
    
    row = pdf.iloc[0]
    
    print("="*80)
    print(f"🔍 PREDICTION FOR {ticker}")
    print("="*80)
    print(f"\nPrediction Date: {row['prediction_date']}")
    print(f"Predicted For: {row['predicted_for_date']}")
    print(f"\nSignal: {row['signal']}")
    print(f"Prediction: {row['prediction']} ({'TĂNG' if row['prediction'] == 1 else 'GIẢM'})")
    print(f"Probability: {row['probability']:.1%}")
    print(f"Confidence: {row['confidence']}")
    print(f"\nLatest Data: {row['latest_data_date']}")
    print("="*80)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Show summary
    show_latest_summary()
    
    # Query examples
    print("\n\n📋 QUERY EXAMPLES:")
    print("-"*80)
    
    # Example 1: Latest predictions
    print("\n1. Get latest predictions (all tickers):")
    print("   df = get_latest_predictions()")
    print("   df.show(10)")
    
    # Example 2: Specific ticker
    print("\n2. Get prediction for specific ticker:")
    print("   df = get_predictions_for_ticker('VCB')")
    print("   df.show()")
    
    # Example 3: Filter by signal
    print("\n3. Get all TĂNG predictions:")
    print("   df = get_predictions_by_signal(1)  # 1 = TĂNG")
    print("   df.show(20)")
    
    # Example 4: High confidence only
    print("\n4. Get HIGH confidence predictions:")
    print("   df = get_high_confidence_predictions()")
    print("   df.show()")
    
    # Example 5: History
    print("\n5. Get prediction history for VCB (last 30 days):")
    print("   df = get_predictions_history('VCB', days=30)")
    print("   df.show()")
    
    # Example 6: Helper functions
    print("\n6. Quick helpers:")
    print("   show_latest_summary()  # Show summary")
    print("   find_ticker('VCB')     # Find specific ticker")
    
    print("-"*80)
