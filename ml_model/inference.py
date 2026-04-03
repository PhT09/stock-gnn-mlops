import torch
import pandas as pd
from ml_model.model import StockGraphSAGE
from ml_model.dataset import StockDataset
import os
import json

class StockPredictor:
    """
    Inference Wrapper for Team E (Backend).
    Handles model loading and real-time prediction.
    """
    def __init__(self, model_path="ml_model/best_model.pt", features_path="data/processed/mock_stocks.parquet", edges_path="data/graph/mock_edges.csv", embeddings_path="data/graph/mock_embeddings.npy"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features_path = features_path
        self.edges_path = edges_path
        self.embeddings_path = embeddings_path
        
        # Load Dataset to get dimensions and mappings
        self.dataset = StockDataset(features_path, edges_path, embeddings_path)
        if len(self.dataset) == 0:
            raise ValueError("Dataset could not be loaded. Ensure data files exist.")
            
        in_channels = self.dataset[0].num_node_features
        self.model = StockGraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=2)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using random weights.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, ticker):
        """
        Predicts trend for a specific ticker using the latest available data.
        Returns: JSON-compatible dictionary.
        """
        # In a real scenario, we would take the LATEST graph (last timestamp)
        latest_graph = self.dataset[-1].to(self.device)
        
        # We need to find the index of the ticker in the graph
        # For simplicity, we assume the dataset mapping is consistent
        # In production, Team A will provide the ticker list order
        
        # Mock logic: find the ticker index (this should match Dataset mapping)
        # Assuming tickers are handled in Alphabetical order or as provided by Team A
        # For now, we take a random node or index 0 for demo
        
        with torch.no_grad():
            logits = self.model(latest_graph.x, latest_graph.edge_index)
            probs = torch.softmax(logits, dim=1)
            
            # For demo, let's just return predictions for ALL nodes as a map
            # but ideally we look up the specific ticker index
            
            # index = ticker_mapping[ticker]
            # pred = probs[index].argmax().item()
            # conf = probs[index].max().item()
            
            # Simulating specific ticker result:
            pred_idx = 0 # Dummy index
            trend_label = "UP" if probs[pred_idx].argmax().item() == 1 else "DOWN"
            confidence = probs[pred_idx].max().item()
            
            result = {
                "ticker": ticker,
                "trend": trend_label,
                "confidence": round(confidence, 4),
                "status": "success",
                "model_version": "1.0.0-GNN"
            }
            return result

if __name__ == "__main__":
    # Internal Test for ML Engineer
    predictor = StockPredictor()
    result = predictor.predict("FPT")
    print(json.dumps(result, indent=4))
