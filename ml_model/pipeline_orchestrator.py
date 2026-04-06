import os
import json
import torch
from ml_model.trainer import run_experiment
import shutil

METADATA_PATH = "ml_model/metadata.json"
MODEL_PATH = "ml_model/best_model.pt"

def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {"best_f1": 0.0, "version": 0}

def save_metadata(metadata):
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

def run_pipeline(features, edges, embeddings, epochs=10):
    """
    Orchestrates the ML Pipeline:
    1. Runs training.
    2. Compares F1 score with existing best.
    3. Keeps or replaces the best model.
    """
    print("--- Starting ML Pipeline Orchestration ---")
    
    # 1. Load current metadata
    metadata = load_metadata()
    print(f"Current Best F1: {metadata['best_f1']:.4f} (Version: {metadata['version']})")
    
    # 2. Run new training session
    # Temporarily save the new model to a different file
    temp_model_path = "ml_model/latest_attempt.pt"
    
    # We need to modify trainer.py slightly to save to a specific path, 
    # but for now, it saves to best_model.pt. 
    # Let's backup the old best_model.pt first.
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, "ml_model/prev_best_model.pt")

    results = run_experiment(features, edges, embeddings, epochs=epochs)
    
    new_f1 = results["f1"]
    print(f"New Training F1: {new_f1:.4f}")
    
    # 3. Compare and Decide
    if new_f1 > metadata["best_f1"]:
        print(">>> SUCCESS: New model is better. Updating best_model.pt.")
        metadata["best_f1"] = new_f1
        metadata["version"] += 1
        save_metadata(metadata)
        # best_model.pt is already saved by trainer.py
    else:
        print(">>> NOTICE: New model did not improve. Reverting to previous best.")
        if os.path.exists("ml_model/prev_best_model.pt"):
            shutil.copy("ml_model/prev_best_model.pt", MODEL_PATH)
            
    print("--- Pipeline Orchestration Finished ---")

if __name__ == "__main__":
    feat = "data/processed/mock_stocks.parquet"
    edge = "data/graph/mock_edges.csv"
    emb = "data/graph/mock_embeddings.npy"
    
    run_pipeline(feat, edge, emb, epochs=3)
