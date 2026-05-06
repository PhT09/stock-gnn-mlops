import sys
import os
sys.path.append('/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops')

from data_engineering.ingestion import ingest_data
from data_engineering.preprocessing import clean_data
from data_engineering.feature_engineering import engineer_features
from ml_model.train import train

def run_pipeline(force_train=False):
    print("--- Starting MLOps Pipeline ---")
    
    # 1. Ingestion with metadata
    metadata = ingest_data(check_freshness=True, force=force_train)
    
    if not metadata.get("success"):
        print(f"❌ Pipeline stopped due to ingestion failure: {metadata.get('error')}")
        return
    
    # Print ingestion summary
    print(f"\n📊 DATA SUMMARY:")
    print(f"  Latest date: {metadata.get('latest_date')}")
    print(f"  Date range: {metadata.get('oldest_date')} to {metadata.get('latest_date')}")
    print(f"  Total rows: {metadata.get('total_rows'):,}")
    print(f"  Data age: {metadata.get('days_old')} days old")
    
    if metadata.get('warnings'):
        for warning in metadata['warnings']:
            print(f"  {warning}")
    
    # 2. Check if we have new data
    has_new_data = metadata.get('has_new_data', False)
    
    if not has_new_data and not force_train:
        print("\n⏸️  PIPELINE PAUSED: Không có data mới")
        print(f"   Lần xử lý cuối: {metadata.get('last_processed_at')}")
        print(f"   Dùng run_pipeline(force_train=True) để bắt buộc train")
        return
    
    print("\n🚀 TIẾP TỤC PIPELINE: Phát hiện data mới hoặc force mode")
    
    from mlops.reporter import generate_report
    # 3. Training
    model, df, metrics = train()
    
    # 4. Report & Insights & Email
    generate_report(model, df, metrics, data_metadata=metadata)
    
    print("\n--- Pipeline Completed Successfully ---")
    print(f"✅ Trained on {metadata.get('total_rows'):,} samples")
    print(f"✅ Data up to: {metadata.get('latest_date')}")
    print(f"✅ Model metrics: Accuracy={metrics.get('accuracy', 0):.3f}, F1={metrics.get('f1_score', 0):.3f}")

if __name__ == "__main__":
    run_pipeline()
