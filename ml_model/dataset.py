import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Dataset
import os

class StockDataset(Dataset):
    """
    Tập dữ liệu PyTorch Geometric tùy chỉnh cho mô hình Chứng khoán GNN.
    
    Yêu cầu cấu trúc dữ liệu từ Người A và Người B:
    - features_path: Parquet/CSV file chứa [ticker, date, open, high, low, close, volume, ma_5, ma_10, volatility, return...]
    - edge_index_path: CSV file chứa [source, target, weight] (từ mạng lưới tương quan - Người B)
    - embeddings_path: NPY file chứa Node2Vec embeddings từ Người B.
    - centrality_path: (Tùy chọn) CSV file chứa mức độ trung tâm (Degree, PageRank).
    """
    def __init__(self, features_path, edge_index_path, embeddings_path=None, centrality_path=None, transform=None, pre_transform=None):
        super(StockDataset, self).__init__(None, transform, pre_transform)
        self.features_path = features_path
        self.edge_index_path = edge_index_path
        self.embeddings_path = embeddings_path
        self.centrality_path = centrality_path
        
        # Load và xử lý dữ liệu ngay khi khởi tạo
        self.data_list = self._process_data()

    def _process_data(self):
        """
        Xử lý các file thô thành danh sách các đối tượng đồ thị `torch_geometric.data.Data`.
        - Mỗi Snapshot (1 thời điểm 15 phút / 1 ngày) là một Đồ thị (Graph Data).
        """
        # ---------------------------------------------------------
        # 1. LOAD NODE FEATURES (Dữ liệu từ Người A)
        # ---------------------------------------------------------
        if not os.path.exists(self.features_path):
            print(f"Warning: Không tìm thấy file Features {self.features_path}.")
            return []
            
        if self.features_path.endswith('.parquet'):
            df = pd.read_parquet(self.features_path)
        else:
            df = pd.read_csv(self.features_path)
            
        # Chuẩn hóa tên cột
        if 'ticket' in df.columns:
            df = df.rename(columns={'ticket': 'ticker'})
            
        # Xử lý Label (Nếu chưa có nhãn Tăng/Giảm, tự sinh từ logic Giá đóng > Giá mở)
        if 'label' not in df.columns:
            print("Đang tự động sinh nhãn (Label) cho dự đoán: 1 (Tăng) / 0 (Giảm)...")
            if 'close' in df.columns and 'open' in df.columns:
                df['label'] = (df['close'] > df['open']).astype(int)
            else:
                df['label'] = 0 # Dummy fallback
            
        tickers = sorted(df['ticker'].unique())
        ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
        
        # ---------------------------------------------------------
        # 2. LOAD GRAPH STRUCTURE (Dữ liệu Cạnh từ Người B)
        # ---------------------------------------------------------
        if not os.path.exists(self.edge_index_path):
            print(f"Warning: Không tìm thấy file Edges {self.edge_index_path}.")
            return []
            
        edge_df = pd.read_csv(self.edge_index_path)
        
        # Khớp (Reconcile) mã cổ phiếu với Index Node trên ma trận
        def reconcile_ticker(t):
            if t in ticker_to_idx: return ticker_to_idx[t]
            if str(t).startswith('STOCK_'):
                idx = int(str(t).split('_')[1])
                if idx < len(tickers): return ticker_to_idx[tickers[idx]]
            return None

        valid_edges = []
        valid_weights = []
        for _, row in edge_df.iterrows():
            s_idx = reconcile_ticker(row['source'])
            t_idx = reconcile_ticker(row['target'])
            weight = row.get('weight', 1.0)
            
            if s_idx is not None and t_idx is not None:
                valid_edges.append([s_idx, t_idx])
                valid_weights.append(weight)
        
        if not valid_edges:
            print("Warning: Edges rỗng sau khi khớp số liệu.")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float)
        else:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(valid_weights, dtype=torch.float)
        
        # ---------------------------------------------------------
        # 3. LOAD EXTRA GRAPH FEATURES (Người B: Node Embeddings & Centrality)
        # ---------------------------------------------------------
        embeddings = None
        if self.embeddings_path and os.path.exists(self.embeddings_path):
            try: embeddings = np.load(self.embeddings_path)
            except: embeddings = pd.read_csv(self.embeddings_path).values
                
        centrality_df = None
        if self.centrality_path and os.path.exists(self.centrality_path):
            centrality_df = pd.read_csv(self.centrality_path)
            if 'ticker' not in centrality_df.columns:
                centrality_df['ticker'] = tickers[:len(centrality_df)]

        # ---------------------------------------------------------
        # 4. CHIA THÀNH CÁC GRAPH SNAPSHOTS THEO THỜI GIAN
        # ---------------------------------------------------------
        graphs = []
        dates = sorted(df['date'].unique()) if 'date' in df.columns else [None]
        
        for date in dates:
            if date is not None:
                day_df = df[df['date'] == date].set_index('ticker').reindex(tickers)
            else:
                day_df = df.set_index('ticker').reindex(tickers)
                
            # Điền 0 cho các đỉnh (cổ phiếu) bị thiếu dữ liệu trong phiên đó
            day_df = day_df.fillna(0)
            
            # Lọc Node Features X (Loại bỏ các trường id, label, date)
            exclude = ['ticker', 'date', 'label']
            x_cols = [c for c in day_df.columns if c not in exclude]
            x_feat = day_df[x_cols].values
            
            # Gộp Centrality (Nếu có)
            if centrality_df is not None:
                day_cent = centrality_df.set_index('ticker').reindex(tickers).fillna(0)
                cent_cols = [col for col in day_cent.columns if col not in exclude]
                x_feat = np.hstack([x_feat, day_cent[cent_cols].values])
                
            x_tensor = torch.tensor(x_feat, dtype=torch.float)
            
            # Gộp Embedding Node2Vec (Nếu có)
            if embeddings is not None:
                emb_t = torch.tensor(embeddings[:len(tickers)], dtype=torch.float)
                # Pad nếu số lượng embed rỗng khác lượng tickers
                if emb_t.size(0) < x_tensor.size(0):
                    padding = torch.zeros((x_tensor.size(0) - emb_t.size(0), emb_t.size(1)))
                    emb_t = torch.cat([emb_t, padding], dim=0)
                x_tensor = torch.cat([x_tensor, emb_t], dim=-1)
                
            # Target Labels Y
            y_tensor = torch.tensor(day_df['label'].values, dtype=torch.long)
            
            # Khởi tạo đối tượng Đồ thị PyTorch Geometric
            data_snapshot = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_weight, y=y_tensor)
            if date is not None:
                data_snapshot.date = date
            graphs.append(data_snapshot)
            
        print(f"✅ Đã tải thành công {len(graphs)} Graph Snapshots.")
        return graphs

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

if __name__ == "__main__":
    # Test file nội bộ dưới local
    import sys
    print("Mô-đun Dataset đã khởi tạo. Hãy gọi class StockDataset từ model trainer của bạn.")
