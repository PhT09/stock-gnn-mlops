import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from ml_model.evaluate import evaluate_model
import os
import json
import shutil

MODEL_DIR = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.json")
PREV_BEST_MODEL_PATH = os.path.join(MODEL_DIR, "prev_best_model.json")
METRICS_PATH = os.path.join(MODEL_DIR, "best_metrics.json")

def load_best_metrics():
    """Load metrics của best model hiện tại"""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return None

def save_model_files(model, metrics, model_info):
    """Lưu model files với version control"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Backup best model cũ thành prev_best_model (nếu có)
    if os.path.exists(BEST_MODEL_PATH):
        print(f"📦 Backing up current best model → prev_best_model.json")
        shutil.copy(BEST_MODEL_PATH, PREV_BEST_MODEL_PATH)
    
    # 2. Lưu model mới làm best_model
    print(f"💾 Saving new best model → best_model.json")
    model.save_model(BEST_MODEL_PATH)
    
    # 3. Lưu metrics
    metrics_data = {
        "metrics": metrics,
        "model_info": model_info,
        "saved_at": pd.Timestamp.now().isoformat()
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"✅ Model files saved:")
    print(f"   - Best: {BEST_MODEL_PATH}")
    print(f"   - Previous: {PREV_BEST_MODEL_PATH}")
    print(f"   - Metrics: {METRICS_PATH}")

def compare_models(new_metrics, old_metrics):
    """
    So sánh model mới với model cũ.
    Return True nếu model mới tốt hơn.
    """
    if old_metrics is None:
        print("\n🆕 Đây là model đầu tiên → Tự động lưu làm best model")
        return True
    
    print("\n📊 SO SÁNH MODEL MỚI VỚI BEST MODEL:")
    print("-" * 60)
    print(f"{'Metric':<15} {'Old Best':<12} {'New':<12} {'Improve':<10}")
    print("-" * 60)
    
    # Primary metric: AUC-ROC
    old_auc = old_metrics['metrics']['auc_roc']
    new_auc = new_metrics['auc_roc']
    auc_improve = new_auc - old_auc
    auc_better = new_auc > old_auc
    
    print(f"{'AUC-ROC':<15} {old_auc:<12.4f} {new_auc:<12.4f} {auc_improve:+.4f} {'✅' if auc_better else '❌'}")
    
    # Secondary metrics
    for metric in ['accuracy', 'f1_score']:
        old_val = old_metrics['metrics'][metric]
        new_val = new_metrics[metric]
        improve = new_val - old_val
        better = new_val > old_val
        print(f"{metric.capitalize():<15} {old_val:<12.4f} {new_val:<12.4f} {improve:+.4f} {'✅' if better else '❌'}")
    
    print("-" * 60)
    
    # Decision logic: Chủ yếu dựa vào AUC-ROC
    if auc_better:
        print(f"\n✅ MODEL MỚI TỐT HƠN: AUC-ROC tăng {auc_improve:+.4f}")
        return True
    else:
        print(f"\n❌ MODEL MỚI KHÔNG TỐT HƠN: AUC-ROC giảm {auc_improve:.4f}")
        print(f"   → Giữ nguyên best model hiện tại")
        return False

def train(data_path="data/raw/stock_data"):
    print("Loading feature table from Databricks...")
    df = pd.read_parquet(data_path)
    
    # Trích xuất mảng values từ Spark ML DenseVector dictionary
    import numpy as np
    X = np.vstack(df['scaled_features'].apply(lambda x: x['values']).values)
    y = df['target']
    
    # Time-based split (e.g., last 20% is test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Kiểm tra xem có đang chạy trên Databricks không
    is_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
    
    if not is_databricks:
        # Chạy ở Local
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"file:///{os.path.abspath('mlruns').replace(os.sep, '/')}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Stock_Prediction_XGBoost")
    else:
        # Chạy trên Databricks
        mlflow.set_experiment("/Shared/Stock_Prediction_XGBoost")
    
    with mlflow.start_run() as run:
        params = {
            "objective": "binary:logistic",
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 100
        }
        mlflow.log_params(params)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Model info
        model_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "params": params
        }
        
        # ============================================================
        # SO SÁNH VỚI BEST MODEL VÀ LƯU FILE
        # ============================================================
        
        old_metrics = load_best_metrics()
        is_better = compare_models(metrics, old_metrics)
        
        if is_better:
            # Lưu model mới làm best model
            save_model_files(model, metrics, model_info)
            
            # Log artifacts vào MLflow
            mlflow.log_artifact(BEST_MODEL_PATH, "model_files")
            if os.path.exists(PREV_BEST_MODEL_PATH):
                mlflow.log_artifact(PREV_BEST_MODEL_PATH, "model_files")
            mlflow.log_artifact(METRICS_PATH, "model_files")
        else:
            print("\n⏸️  Model không tốt hơn → Không cập nhật best_model.json")
        
        # ============================================================
        # LOG VÀO MLFLOW + UNITY CATALOG (như cũ)
        # ============================================================
        
        input_example = pd.DataFrame(X_train[:5])
        
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="xgboost-model",
            registered_model_name="stock_predictor",
            input_example=input_example
        )
        
        print("\nModel training completed and logged to MLflow.")
        return model, df, metrics

if __name__ == "__main__":
    train()
