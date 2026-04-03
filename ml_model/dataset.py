import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
import os

class StockDataset(Dataset):
    """
    Custom Dataset for loading Stock Market Graph data.
    Expected data structure:
    - features_path: Parquet file with [ticker, date, feat1, feat2, ..., label]
    - edge_index_path: CSV file with [source, target, weight]
    - embeddings_path: NPY file or CSV with Node2Vec embeddings
    """
    def __init__(self, features_path, edge_index_path, embeddings_path=None, transform=None, pre_transform=None):
        super(StockDataset, self).__init__(None, transform, pre_transform)
        self.features_path = features_path
        self.edge_index_path = edge_index_path
        self.embeddings_path = embeddings_path
        
        self.data_list = self._process_data()

    def _process_data(self):
        """
        Process raw files into a list of torch_geometric.data.Data objects.
        For stock prediction, each timestamp can be a single graph.
        """
        # 1. Load Features
        if not os.path.exists(self.features_path):
            print(f"Warning: Features path {self.features_path} not found.")
            return []
            
        df = pd.read_parquet(self.features_path)
        
        # 2. Load Edge Index (Static Topology for now)
        if not os.path.exists(self.edge_index_path):
            print(f"Warning: Edge index path {self.edge_index_path} not found.")
            return []
            
        edge_df = pd.read_csv(self.edge_index_path)
        
        # Mapping Tickers to Indices
        tickers = df['ticker'].unique()
        ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
        
        # Convert edge list to indices
        edge_index = torch.tensor([
            [ticker_to_idx[s] for s in edge_df['source']],
            [ticker_to_idx[t] for t in edge_df['target']]
        ], dtype=torch.long)
        
        # 3. Load Embeddings (Optional)
        embeddings = None
        if self.embeddings_path and os.path.exists(self.embeddings_path):
            embeddings = np.load(self.embeddings_path) # Assume NPY
            
        # 4. Create Graph per Timestamp
        graphs = []
        dates = sorted(df['date'].unique())
        
        for date in dates:
            day_df = df[df['date'] == date].set_index('ticker').reindex(tickers)
            
            # Node features (X)
            # Exclude ticker, date, and label
            x_cols = [c for c in day_df.columns if c not in ['ticker', 'date', 'label']]
            x = torch.tensor(day_df[x_cols].values, dtype=torch.float)
            
            # If embeddings are provided, concatenate them to node features
            if embeddings is not None:
                # Mock logic for embeddings alignment: 
                # Embeddings should be [num_nodes, emb_dim]
                emb_tensor = torch.tensor(embeddings, dtype=torch.float)
                x = torch.cat([x, emb_tensor], dim=-1)
                
            # Labels (Y)
            y = torch.tensor(day_df['label'].values, dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data.date = date
            graphs.append(data)
            
        return graphs

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

if __name__ == "__main__":
    # Test script for local verification
    print("StockDataset class initialized.")
