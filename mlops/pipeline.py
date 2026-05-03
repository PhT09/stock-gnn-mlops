import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_engineering.ingestion import ingest_data
from data_engineering.preprocessing import clean_data
from data_engineering.feature_engineering import engineer_features
from ml_model.train import train

def run_pipeline():
    print("--- Starting MLOps Pipeline ---")
    
    # 1. Ingestion
    success = ingest_data()
    if success is False:
        print("Pipeline stopped due to ingestion failure.")
        return
    
    from mlops.reporter import generate_report
    # 2. Training (Databricks đã làm sẵn Feature Engineering)
    model, df, metrics = train()
    
    # 3. Report & Insights & Email
    generate_report(model, df, metrics)
    
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    run_pipeline()
