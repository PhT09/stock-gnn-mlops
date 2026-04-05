import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from ml_model.dataset import StockDataset
from ml_model.model import StockGraphSAGE
import os

def evaluate_model(model_path, features_path, edges_path, embeddings_path):
    """
    Evaluates a trained GNN model on the test dataset.
    """
    # 1. Load Dataset
    dataset = StockDataset(features_path, edges_path, embeddings_path)
    if len(dataset) == 0:
        return
        
    # Split (Same split as trainer)
    train_size = int(0.8 * len(dataset))
    test_dataset = dataset[train_size:]
    
    # 2. Load Model
    in_channels = dataset[0].num_node_features
    model = StockGraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=2)
    
    # Load state dict
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found. Running with initial weights for evaluation demo.")
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 3. Inference
    with torch.no_grad():
        for data in test_dataset:
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.tolist())
            all_labels.extend(data.y.tolist())
            
    # 4. Metrics
    print("\n--- Evaluation Report ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1-Score: {f1_score(all_labels, all_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    # Test path (Mock data)
    feat = "data/raw/stock_data - stock_data.csv"
    edge = "data/graph/mock_edges.csv"
    emb = "data/graph/mock_embeddings.npy"
    
    # Assuming trainer saved a baseline model
    model_file = "ml_model/best_model.pt"
    evaluate_model(model_file, feat, edge, emb)
