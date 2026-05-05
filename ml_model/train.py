import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from ml_model.evaluate import evaluate_model
import os

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
        
        # ✅ FIX: Thêm input_example để MLflow tự động infer signature
        # Unity Catalog yêu cầu signature cho tất cả models
        input_example = pd.DataFrame(X_train[:5])  # Lấy 5 mẫu đầu tiên
        
        # Register the model with signature
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="xgboost-model",
            registered_model_name="stock_predictor",
            input_example=input_example  # ✅ Thêm dòng này
        )
        
        print("Model training completed and logged to MLflow.")
        return model, df, metrics

if __name__ == "__main__":
    train()
