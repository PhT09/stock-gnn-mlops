import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import os

def calculate_pearson_correlation(df, threshold=0.6):
    """
    Calculates Pearson correlation between stock tickers based on their features (e.g., feat1).
    Returns an edge list (source, target, weight).
    """
    # Pivot the data to have dates as index and tickers as columns for a specific feature
    # Using 'feat1' as a proxy for price/return
    pivot_df = df.pivot(index='date', columns='ticker', values='feat1')
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.corr(method='pearson')
    
    edges = []
    tickers = corr_matrix.columns
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            weight = corr_matrix.iloc[i, j]
            if abs(weight) > threshold:
                edges.append({
                    "source": tickers[i],
                    "target": tickers[j],
                    "weight": weight
                })
    
    return pd.DataFrame(edges)

def generate_graph_features(output_dir="data/graph", features_path="data/processed/mock_stocks.parquet"):
    """
    Full pipeline for Graph Engineering:
    1. Pearson Correlation -> Edge list
    2. NetworkX -> Centrality
    3. Node2Vec -> Embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found.")
        return
    
    df = pd.read_parquet(features_path)
    
    # 1. Pearson Correlation
    print("Calculating Pearson Correlation...")
    edge_df = calculate_pearson_correlation(df)
    edge_index_path = os.path.join(output_dir, "edges.csv")
    edge_df.to_csv(edge_index_path, index=False)
    print(f"Saved {len(edge_df)} edges to {edge_index_path}")
    
    # 2. Build NetworkX Graph
    G = nx.from_pandas_edgelist(edge_df, source='source', target='target', edge_attr='weight')
    
    # Fill in isolated nodes (tickers with no edges)
    all_tickers = df['ticker'].unique()
    G.add_nodes_from(all_tickers)
    
    # 3. Centrality Measures
    print("Calculating Centrality measures...")
    degree_cent = nx.degree_centrality(G)
    eigen_cent = nx.eigenvector_centrality(G, max_iter=1000)
    
    centrality_df = pd.DataFrame([
        {"ticker": t, "degree_cent": degree_cent.get(t, 0), "eigen_cent": eigen_cent.get(t, 0)}
        for t in all_tickers
    ])
    centrality_path = os.path.join(output_dir, "centrality.csv")
    centrality_df.to_csv(centrality_path, index=False)
    print(f"Saved centrality measures to {centrality_path}")
    
    # 4. Node2Vec Embeddings
    print("Running Node2Vec (this may take a few seconds)...")
    # p=1, q=1 (standard DeepWalk), dimensions=32
    n2v = Node2Vec(G, dimensions=32, walk_length=10, num_walks=40, workers=1)
    model = n2v.fit(window=5, min_count=1)
    
    # Save embeddings in ticker order
    embeddings = np.array([model.wv[ticker] for ticker in all_tickers])
    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings array to {embeddings_path}")

if __name__ == "__main__":
    generate_graph_features()
