import sys
import os
if "__file__" not in globals():
    project_root = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops"
else:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

from ml_model.train import train
from ml_model.predict_multi_day import predict_multi_day
from mlops.reporter import generate_report

from pyspark.sql import SparkSession

def run_pipeline():
    """
    MLOps Pipeline - TRAIN, PREDICT 15 DAYS & EMAIL
    
    Pipeline này:
    1. Đọc PROCESSED từ /Volumes/workspace/default/stock_data/processed/
    2. Train XGBoost model
    3. ✨ DỰ ĐOÁN TẤT CẢ MÃ cho 15 NGÀY tiếp theo
    4. Gửi email report với CSV 15 ngày
    5. Lưu kết quả dự đoán vào bảng phục vụ API backend
    
    Thời gian: ~4-6 phút
    """
    print("=" * 80)
    print("🚀 Starting MLOps Pipeline")
    print("   Mode: TRAIN, PREDICT 15 DAYS & EMAIL")
    print("=" * 80)
    
    # Step 1: Model Training
    print("\n🤖 Step 1: Model Training")
    print("   📂 Reading from: /Volumes/workspace/default/stock_data/processed/stock_features.parquet")

    try:
        model, df, metrics = train()
        
        print("\n✅ Training completed!")
        print(f"   📊 Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"   📊 F1 Score: {metrics.get('f1_score', 0):.3f}")
        print(f"   📊 AUC-ROC:  {metrics.get('auc_roc', 0):.3f}")
        
    except Exception as e:
        print(f"\n❌ Training FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Step 2: ✨ PREDICTION - DỰ ĐOÁN 15 NGÀY TIẾP THEO
    print("\n🔮 Step 2: Predict Next 15 Trading Days")
    try:
        predictions_df = predict_multi_day(n_days=15)
        
        print("\n✅ Predictions completed!")
        print(f"   📊 Total tickers: {len(predictions_df)}")
        
        # Day 1 summary
        day1_up = (predictions_df['day_1_prediction'] == 1).sum()
        day1_down = (predictions_df['day_1_prediction'] == 0).sum()
        day1_date = predictions_df['day_1_date'].iloc[0]
        
        print(f"   📅 Day 1 ({day1_date}):")
        print(f"      📈 TĂNG: {day1_up} tickers")
        print(f"      📉 GIẢM: {day1_down} tickers")
        
        # Week summary
        print(f"\n   📅 Week overview (Days 1-5):")
        for day in range(1, 6):
            pred_col = f'day_{day}_prediction'
            date_col = f'day_{day}_date'
            up = (predictions_df[pred_col] == 1).sum()
            date = predictions_df[date_col].iloc[0]
            print(f"      {date}: {up}/{len(predictions_df)} TĂNG")
        
        # Step 2b: Save predictions to backend table
        print("\n🗃️ Step 2b: Saving predictions to backend table")
        try:
            spark = SparkSession.builder.getOrCreate()
            pred_spark_df = spark.createDataFrame(predictions_df)
            pred_spark_df.write.mode('overwrite').saveAsTable('workspace.default.stock_predictions')
            print("✅ Predictions saved to workspace.default.stock_predictions")
        except Exception as e:
            print("⚠️  Failed to save predictions to table:", str(e))
            import traceback
            traceback.print_exc()
            # Saving failure is not critical, continue
        
    except Exception as e:
        print(f"\n⚠️  Prediction failed (non-critical): {str(e)}")
        import traceback
        traceback.print_exc()
        predictions_df = None
        # Prediction failure is not critical, continue
    
    # Step 3: Report & Email
    print("\n📧 Step 3: Generate Report & Send Email")
    try:
        from mlops.reporter import send_email
        
        # Generate report (returns html, image, csv paths)
        html_content, image_path, csv_path = generate_report(model, df, metrics, predictions_df)
        
        # Actually send the email!
        subject = f"📊 Stock Predictions Report - {predictions_df['day_1_date'].iloc[0] if predictions_df is not None else 'N/A'}"
        email_sent = send_email(subject, html_content, image_path, csv_path)
        
        if email_sent:
            print("\n✅ Email sent successfully!")
        else:
            print("\n⚠️  Email failed to send (check .env config)")
        
    except Exception as e:
        print(f"\n⚠️  Report generation failed (non-critical): {str(e)}")
        import traceback
        traceback.print_exc()
        # Email failure is not critical, don't raise
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED")
    print("=" * 80)
    print(f"📊 Training Metrics:")
    print(f"   • Accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"   • F1 Score: {metrics.get('f1_score', 0):.3f}")
    print(f"   • AUC-ROC: {metrics.get('auc_roc', 0):.3f}")
    
    if predictions_df is not None and len(predictions_df) > 0:
        print(f"\n🔮 Predictions:")
        print(f"   • Total: {len(predictions_df)} tickers")
        print(f"   • Days: 15 ngày giao dịch tiếp theo")
        print(f"   • From: {predictions_df['day_1_date'].iloc[0]}")
        print(f"   • To: {predictions_df['day_15_date'].iloc[0]}")
        
        # Overall stats
        print(f"\n   📊 Day 1 summary:")
        day1_up = (predictions_df['day_1_prediction'] == 1).sum()
        print(f"      📈 TĂNG: {day1_up} ({day1_up/len(predictions_df)*100:.1f}%)")
        print(f"      📉 GIẢM: {len(predictions_df) - day1_up} ({(1 - day1_up/len(predictions_df))*100:.1f}%)")
        
        # High confidence count
        high_conf = (predictions_df['day_1_confidence'] == 'HIGH').sum()
        print(f"      🎯 HIGH confidence: {high_conf} tickers")
    
    print(f"\n📍 MLflow Experiment: /Shared/Stock_Prediction_XGBoost")
    print(f"📍 CSV Report: data/predictions_15_days.csv")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
