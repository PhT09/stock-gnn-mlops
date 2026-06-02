import sys
import os

# Thêm project root vào path
if "__file__" not in globals():
    project_root = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops"
else:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

from data_engineering.ingestion_databricks import ingest_data
from ml_model.train import train
from mlops.reporter import generate_report

def run_pipeline_databricks(force_train=False, recent_days=None):
    """
    MLOps Pipeline cho Databricks environment.
    
    Args:
        force_train (bool): Bắt buộc train dù data không đổi
        recent_days (int): Chỉ train trên N ngày gần nhất (None = tất cả)
    """
    print("="*80)
    print("🚀 STARTING MLOPS PIPELINE (Databricks)")
    print("="*80)
    
    # 1. Ingestion
    print("\n[STEP 1/3] Data Ingestion...")
    metadata = ingest_data(
        output_folder="/Volumes/workspace/default/stock_data/processed/stock_features.parquet",
        recent_days=recent_days,
        check_freshness=True,
        force=force_train
    )
    
    if not metadata.get("success"):
        print(f"\n❌ PIPELINE FAILED: {metadata.get('error')}")
        return metadata
    
    # Print summary
    print(f"\n📊 DATA SUMMARY:")
    print(f"  Latest date: {metadata.get('latest_date')}")
    print(f"  Total rows: {metadata.get('total_rows'):,}")
    print(f"  Data age: {metadata.get('days_old')} days")
    print(f"  Has new data: {metadata.get('has_new_data')}")
    
    # 2. Check if should train
    has_new_data = metadata.get('has_new_data', False)
    
    if not has_new_data and not force_train:
        print(f"\n⏸️  PIPELINE PAUSED: Không có data mới")
        print(f"   Lần xử lý cuối: {metadata.get('last_processed_at')}")
        print(f"   → Dùng force_train=True để bắt buộc train")
        return {"status": "skipped", "reason": "no_new_data", "metadata": metadata}
    
    print(f"\n🚀 CONTINUING PIPELINE...")
    if force_train:
        print("   Reason: Force mode enabled")
    else:
        print("   Reason: New data detected")
    
    # 3. Training
    print(f"\n[STEP 2/3] Model Training...")
    try:
        model, df, metrics = train(
            data_path="/Volumes/workspace/default/stock_data/processed/stock_features.parquet"
        )
        print(f"\n✅ Training completed!")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
        print(f"   F1 Score: {metrics.get('f1_score', 0):.3f}")
        print(f"   AUC-ROC: {metrics.get('auc_roc', 0):.3f}")
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "step": "training", "error": str(e)}
    
    # 4. Report
    print(f"\n[STEP 3/3] Generating Report...")
    try:
        generate_report(model, df, metrics, data_metadata=metadata)
        print(f"\n✅ Report generated!")
    except Exception as e:
        print(f"\n⚠️  Report generation failed: {str(e)}")
        # Non-critical error, continue
    
    # Summary
    print(f"\n" + "="*80)
    print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"="*80)
    print(f"📊 Summary:")
    print(f"  Data date: {metadata.get('latest_date')}")
    print(f"  Samples trained: {metadata.get('total_rows'):,}")
    print(f"  Model accuracy: {metrics.get('accuracy', 0):.3f}")
    print(f"  MLflow experiment: /Shared/Stock_Prediction_XGBoost")
    
    return {
        "status": "success",
        "metadata": metadata,
        "metrics": metrics
    }

if __name__ == "__main__":
    # Run pipeline
    result = run_pipeline_databricks()
    
    print(f"\n🔍 Final Result:")
    import json
    print(json.dumps(result, indent=2, default=str))
