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
    def __init__(self, model_path="ml_model/best_model.pt", features_path="data/raw/stock_data - stock_data.csv", edges_path="data/graph/mock_edges.csv", embeddings_path="data/graph/mock_embeddings.npy"):
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
        
        # Load ticker mapping from dataset (re-read to get alphabetical mapping)
        df = pd.read_csv(features_path) if features_path.endswith('.csv') else pd.read_parquet(features_path)
        if 'ticket' in df.columns: df = df.rename(columns={'ticket': 'ticker'})
        self.tickers = sorted(df['ticker'].unique())
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}

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
        if ticker not in self.ticker_to_idx:
            return {"status": "error", "message": f"Ticker {ticker} not found in model node mapping."}

        latest_graph = self.dataset[-1].to(self.device)
        pred_idx = self.ticker_to_idx[ticker]
        
        with torch.no_grad():
            logits = self.model(latest_graph.x, latest_graph.edge_index)
            probs = torch.softmax(logits, dim=1)
            
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
    # Try a ticker from the CSV (e.g., VIC)
    result = predictor.predict("VIC")
    print(json.dumps(result, indent=4))
