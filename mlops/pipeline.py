import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append('/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops')

from ml_model.train import train
from mlops.reporter import generate_report

def run_pipeline():
    """
    MLOps Pipeline - TRAIN & EMAIL ONLY
    
    ⚠️  KHÔNG cào data! Data đã được cào sẵn vào PROCESSED bởi job khác.
    
    Pipeline này CHỈ:
    1. Đọc PROCESSED có sẵn tại /Volumes/workspace/default/stock_data/processed/
    2. Train XGBoost model
    3. Gửi email report
    
    Thời gian: ~3-4 phút (nhanh hơn nhiều vì không cào data!)
    """
    print("=" * 80)
    print("🚀 Starting MLOps Pipeline")
    print("   Mode: TRAIN & EMAIL ONLY (No data ingestion)")
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
    
    # Step 2: Report & Email
    print("\n📧 Step 2: Generate Report & Send Email")
    try:
        generate_report(model, df, metrics)
        print("\n✅ Email sent successfully!")
        
    except Exception as e:
        print(f"\n⚠️  Report generation failed (non-critical): {str(e)}")
        # Email failure is not critical, don't raise
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED")
    print("=" * 80)
    print(f"📊 Final Metrics:")
    print(f"   • Accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"   • F1 Score: {metrics.get('f1_score', 0):.3f}")
    print(f"   • AUC-ROC: {metrics.get('auc_roc', 0):.3f}")
    print(f"\n📍 MLflow Experiment: /Shared/Stock_Prediction_XGBoost")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
