import sys
import os
sys.path.append('/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops')

from data_engineering.ingestion import ingest_data
from data_engineering.feature_engineering import engineer_features
from ml_model.train import train

def run_pipeline(force_train=False):
    print("--- Starting MLOps Pipeline ---")
    
    # 1. Ingestion with metadata
    ingest_data()  # No parameters needed
    metadata = {}  # Placeholder - ingestion doesn't return metadata
    
    print("\n📊 Ingestion completed")
    
    # Continue with pipeline - ingest_data() already handles freshness check internally
    
    # 3. Feature Engineering - Process raw data to features
    print("\n🔧 Step 2: Feature Engineering")
    engineer_features()
    
    from mlops.reporter import generate_report
    
    # 4. Training
    print("\n🤖 Step 3: Model Training")
    model, df, metrics = train()
    
    # 5. Report & Insights & Email
    print("\n📧 Step 4: Generate Report & Send Email")
    generate_report(model, df, metrics)
    
    print("\n--- Pipeline Completed Successfully ---")
    print(f"✅ Model metrics: Accuracy={metrics.get('accuracy', 0):.3f}, F1={metrics.get('f1_score', 0):.3f}")

if __name__ == "__main__":
    run_pipeline()
