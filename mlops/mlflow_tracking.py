import mlflow
import os

def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to {tracking_uri}")

def transition_model_stage(model_name, version, stage):
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    print(f"Transitioned {model_name} v{version} to {stage}")
