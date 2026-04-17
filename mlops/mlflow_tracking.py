import mlflow
from typing import Dict, Any, Optional

# Lưu ý bảo mật: Chúng ta không import torch ở đây để tránh lỗi WinError 4551.
# Các thư viện này sẽ được import động khi cần thiết.

class MLflowTracker:
    """
    Lớp hỗ trợ theo dõi thí nghiệm ML (ML Tracking) sử dụng MLflow.
    Được thiết kế đặc thù cho dự án dự đoán cổ phiếu GNN.
    """

    def __init__(self, experiment_name: str = "Stock_GNN_Prediction"):
        """
        Khởi tạo Tracker và thiết lập Experiment.
        
        Args:
            experiment_name (str): Tên của thí nghiệm trong MLflow.
        """
        self.experiment_name = experiment_name
        # Thiết lập địa chỉ server local (mặc định port 5000)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        """Bắt đầu một phiên chạy (run) mới."""
        return mlflow.start_run(run_name=run_name)

    def log_parameters(self, params: Dict[str, Any]):
        """
        Ghi lại các siêu tham số (Hyperparameters) của mô hình.
        
        Args:
            params (Dict): Dictionary chứa tên tham số và giá trị (ví dụ: learning_rate).
        """
        mlflow.log_params(params)
        print(f"✅ Đã log {len(params)} tham số.")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Ghi lại các chỉ số đánh giá (Evaluation Metrics).
        
        Args:
            metrics (Dict): Dictionary chứa tên chỉ số và giá trị (ví dụ: f1_score).
            step (int, optional): Epoch hiện tại hoặc bước huấn luyện.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_pytorch_model(self, model, model_name: str, artifact_path: str = "model"):
        """Lưu trữ mô hình PyTorch, nếu lỗi sẽ thông báo (dành cho Role C)."""
        try:
            # Import động mà không gây xung đột scope
            import importlib
            mlflow_pytorch = importlib.import_module("mlflow.pytorch")
            
            mlflow_pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=model_name
            )
            print(f"✅ Đã lưu trữ và đăng ký mô hình: {model_name}")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể log model PyTorch (Giải thích cho Role C): {e}")
            print("💡 MLOps Note: Đang thực hiện log lỗi vào MLflow UI...")
            # Sử dụng mlflow toàn cục để ghi lại lỗi
            mlflow.log_dict({"status": "model_blocked", "error": str(e)}, "model_error.json")

# --- MINI MOCK RUN (Dùng để test thử ngay lập tức) ---
if __name__ == "__main__":
    tracker = MLflowTracker(experiment_name="Stock_GNN_Test_Run")

    print("🚀 Bắt đầu quá trình Mock Run để kiểm tra MLflow Server...")
    
    with tracker.start_run(run_name="MLflow_System_Check"):
        # 1. Log Parameters mẫu
        gnn_params = {
            "learning_rate": 0.001,
            "status": "testing_policy_safe"
        }
        tracker.log_parameters(gnn_params)

        # 2. Log Metrics mẫu
        import random
        for epoch in range(5):
            tracker.log_metrics({"accuracy": 0.5 + (epoch * 0.1)}, step=epoch)
        
        # 3. Log Model (Sẽ thử log, nếu lỗi policy sẽ không làm sập script)
        print("🔍 Đang thử nghiệm tính năng Log Model (có thể bị chặn bởi Windows Policy)...")
        tracker.log_pytorch_model(model=None, model_name="MockModel_Test")

    print("\n🎉 Hệ thống Tracking vẫn hoạt động! Hãy vào http://127.0.0.1:5000 để kiểm tra log.")
