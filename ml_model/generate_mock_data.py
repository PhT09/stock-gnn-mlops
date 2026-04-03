import pandas as pd
import numpy as np
import os
import torch

def generate_mock_data(output_dir="data"):
    """
    Generates mock stock data for testing.
    - data/processed/mock_stocks.parquet
    - data/graph/mock_edges.csv
    - data/graph/mock_embeddings.npy
    """
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "graph"), exist_ok=True)

    tickers = [f"STOCK_{i}" for i in range(10)]
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    
    # 1. Features
    data = []
    for date in dates:
        for ticker in tickers:
            data.append({
                "date": date,
                "ticker": ticker,
                "feat1": np.random.randn(),
                "feat2": np.random.randn(),
                "label": np.random.randint(0, 2)
            })
    
    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(output_dir, "processed", "mock_stocks.parquet"))
    print(f"Generated {len(df)} feature rows.")

    # 2. Edges (Static graph)
    edges = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            if np.random.rand() > 0.6: # 40% probability of correlation
                edges.append({"source": tickers[i], "target": tickers[j], "weight": np.random.rand()})
                
    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(os.path.join(output_dir, "graph", "mock_edges.csv"), index=False)
    print(f"Generated {len(edge_df)} edges.")

    # 3. Embeddings (Mock Node2Vec)
    embeddings = np.random.randn(len(tickers), 32)
    np.save(os.path.join(output_dir, "graph", "mock_embeddings.npy"), embeddings)
    print(f"Generated embeddings for {len(tickers)} nodes.")

if __name__ == "__main__":
    generate_mock_data()
