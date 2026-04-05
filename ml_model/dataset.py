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
            
        if self.features_path.endswith('.parquet'):
            df = pd.read_parquet(self.features_path)
        else:
            df = pd.read_csv(self.features_path)
            
        # Standardize column names
        if 'ticket' in df.columns:
            df = df.rename(columns={'ticket': 'ticker'})
            
        # Ensure label exists (for mock data)
        if 'label' not in df.columns:
            print("Generating dummy labels (Trend: Close > Open)...")
            df['label'] = (df['close'] > df['open']).astype(int)
            
        # 2. Load Edge Index (Static Topology for now)
        if not os.path.exists(self.edge_index_path):
            print(f"Warning: Edge index path {self.edge_index_path} not found.")
            return []
            
        edge_df = pd.read_csv(self.edge_index_path)
        
        # Mapping Tickers to Indices
        tickers = sorted(df['ticker'].unique())
        ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
        
        # Reconciliation: If edges use 'STOCK_N' but df uses real names
        # Map STOCK_N -> tickers[N]
        def reconcile_ticker(t):
            if t in ticker_to_idx:
                return ticker_to_idx[t]
            if t.startswith('STOCK_'):
                idx = int(t.split('_')[1])
                if idx < len(tickers):
                    return ticker_to_idx[tickers[idx]]
            return None

        valid_edges = []
        for _, row in edge_df.iterrows():
            s_idx = reconcile_ticker(row['source'])
            t_idx = reconcile_ticker(row['target'])
            if s_idx is not None and t_idx is not None:
                valid_edges.append([s_idx, t_idx])
        
        if not valid_edges:
            print("Warning: No valid edges found after reconciliation.")
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        
        # 3. Load Embeddings (Optional)
        embeddings = None
        if self.embeddings_path and os.path.exists(self.embeddings_path):
            # Try loading NPY first, then CSV
            try:
                embeddings = np.load(self.embeddings_path)
            except:
                emb_df = pd.read_csv(self.embeddings_path)
                embeddings = emb_df.values
            
        # 4. Create Graph per Timestamp
        graphs = []
        dates = sorted(df['date'].unique())
        
        for date in dates:
            day_df = df[df['date'] == date].set_index('ticker').reindex(tickers)
            # Fill missing nodes with zero features and 0 label
            day_df = day_df.fillna(0)
            
            # Node features (X)
            # Exclude non-feature columns
            exclude = ['ticker', 'date', 'label']
            x_cols = [c for c in day_df.columns if c not in exclude]
            x = torch.tensor(day_df[x_cols].values, dtype=torch.float)
            
            # If embeddings are provided, concatenate them to node features
            if embeddings is not None:
                # Ensure embeddings match the number of current tickers
                emb_tensor = torch.tensor(embeddings[:len(tickers)], dtype=torch.float)
                if emb_tensor.size(0) < x.size(0):
                    # Pad if necessary
                    padding = torch.zeros((x.size(0) - emb_tensor.size(0), emb_tensor.size(1)))
                    emb_tensor = torch.cat([emb_tensor, padding], dim=0)
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
