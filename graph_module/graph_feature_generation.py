"""
Graph Feature Generation Module
--------------------------------
Mục tiêu: Đọc ma trận tương quan Pearson đã tính sẵn, lọc bằng Threshold,
xây dựng Đồ thị (Graph), trích xuất đặc trưng mạng lưới (Centrality) 
và mô phỏng nhúng không gian 64 chiều bằng thuật toán Node2Vec.
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import multiprocessing
import pandas as pd
import numpy as np
import networkx as nx

# Bắt lỗi môi trường nếu thiếu package Node2Vec
try:
    from node2vec import Node2Vec
except ImportError:
    raise ImportError("Vui lòng cài đặt thư viện 'node2vec' bằng lệnh: pip install node2vec")

# 2. LOAD CORRELATION MATRIX
def load_correlation(file_path: str) -> pd.DataFrame:
    """
    Đọc ma trận tương quan từ file CSV.
    Cột đầu tiên mặc định phải được ép thành index (Stock Symbols).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file ma trận: {file_path}")
        
    df_corr = pd.read_csv(file_path, index_col=0)
    
    # Ép kiểu và kiểm tra tính đối xứng / cấu trúc vuông
    if df_corr.shape[0] != df_corr.shape[1]:
        raise ValueError("Ma trận tương quan phải có cấu trúc vuông NxN.")
        
    return df_corr

# 3. THRESHOLD FILTERING + GRAPH CONSTRUCTION
def build_graph(corr_matrix: pd.DataFrame, threshold: float = 0.6) -> nx.Graph:
    """
    Xây dựng đồ thị vô hướng từ ma trận, áp dụng bộ lọc (Threshold).
    Loại bỏ vòng tự thân (self-loops) nơi tương quan đường chéo = 1.0.
    """
    G = nx.Graph()
    tickers = corr_matrix.columns.tolist()
    
    # Nạp toàn bộ các đỉnh (Ngay cả khi không có kết nối để giữ đúng kiến trúc)
    G.add_nodes_from(tickers)
    
    num_nodes = corr_matrix.shape[0]
    
    # Duyệt nửa trên ma trận để nối Edges (Đồ thị vô hướng)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            stock_a = tickers[i]
            stock_b = tickers[j]
            weight = corr_matrix.iloc[i, j]
            
            # Cắt Threshold và đảm bảo giá trị toán học đúng
            if pd.notna(weight) and abs(weight) > threshold:
                G.add_edge(stock_a, stock_b, weight=weight)
                
    return G

# 4. NODE2VEC EMBEDDING
def run_node2vec(G: nx.Graph, dimensions: int = 64) -> tuple:
    """
    Sử dụng Node2Vec tạo ra Embeddings cho các đỉnh (Nodes).
    Trả về: Numpy Array (Embeddings), List (Node Labels để map index)
    """
    # Lấy toàn bộ số lượng CPU core hiện hành trên máy
    cores = multiprocessing.cpu_count()
    print(f"Bắt đầu chạy Node2Vec với {cores} đa luồng CPU...")
    
    node_labels = list(G.nodes())
    
    # Nếu đồ thị trống (Không có Edge), sinh ra zero tensor để chống crash
    if G.number_of_edges() == 0:
        print("CẢNH BÁO: Đồ thị không có nhịp nối. Sinh mô hình nhúng mặc định...")
        return np.zeros((len(node_labels), dimensions)), node_labels
        
    try:
        # Cấu hình sức mạnh phân rã (Walk length = 30, Num walks = 200)
        node2vec_model = Node2Vec(
            G, 
            dimensions=dimensions, 
            walk_length=30, 
            num_walks=200, 
            workers=cores, 
            quiet=True
        )
        
        # Huấn luyện Window context
        model = node2vec_model.fit(window=10, min_count=1, batch_words=4)
        
        # Trích xuất embeddings tuần tự chính xác theo index của G.nodes()
        embeddings = np.array([model.wv[node] for node in node_labels])
        return embeddings, node_labels
        
    except Exception as e:
        print(f"Xảy ra lỗi trong quá trình học Embedding: {e}")
        return np.zeros((len(node_labels), dimensions)), node_labels

# 5. CENTRALITY COMPUTATION
def compute_centrality(G: nx.Graph) -> pd.DataFrame:
    """
    Tính toán các chỉ số mức độ quan trọng (Centrality Metrics) của mạng.
    Bao gồm: Degree, Betweenness, Closeness.
    """
    # Tính Degree (Độ kết nối gốc)
    degree_dict = nx.degree_centrality(G)
    
    # Tính Betweenness (Trung tâm trung gian - con đường ngắn nhất)
    # Cấu hình weight='weight' để chịu ảnh hưởng bởi độ lớn tương quan
    betweenness_dict = nx.betweenness_centrality(G, weight='weight')
    
    # Tính Closeness (Độ gần - lan truyền thông tin)
    closeness_dict = nx.closeness_centrality(G)
    
    # Tập hợp vào DataFrame
    df_centrality = pd.DataFrame({
        'degree_centrality': pd.Series(degree_dict),
        'betweenness_centrality': pd.Series(betweenness_dict),
        'closeness_centrality': pd.Series(closeness_dict)
    })
    
    # Điền index cho có tên cổ phiếu
    df_centrality.index.name = 'stock_symbol'
    return df_centrality

# 6. SAVE OUTPUTS
if __name__ == "__main__":
    
    # Đường dẫn động linh hoạt 
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CORR_PATH = os.path.join(BASE_DIR, "data", "processed", "parallel_correlation_matrix.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Mục 1: Đọc Ma Trận
        print("\n--- GRAPH FEATURE ENGINEERING MODULE ---")
        print("1. Đang tải Ma trận Correlation Matrix liên đới...")
        correlation_matrix = load_correlation(CORR_PATH)
        
        # Mục 2 & 3: Lọc Threshold và Build Graph
        THRESHOLD = 0.6
        print(f"2. Đang xây dựng cấu trúc Mạng Lưới (Threshold = {THRESHOLD})...")
        stock_graph = build_graph(correlation_matrix, threshold=THRESHOLD)
        print(f" -> Cấu trúc đồ thị: {stock_graph.number_of_nodes()} Đỉnh, {stock_graph.number_of_edges()} Cạnh.")
        
        # Mục 4: Khai phá Embbending Node2Vec
        print("\n3. Đang giả lập nén không gian Vector Node2Vec...")
        embeddings_matrix, node_order = run_node2vec(stock_graph, dimensions=64)
        
        # Mục 5: Trích xuất chỉ số Toán Học Centrality
        print("4. Đang phân tích chỉ số đặc tả Centrality Metrics...")
        centrality_df = compute_centrality(stock_graph)
        
        # Mục 6: Đẩy kết quả ra File
        print("\n5. Tập trung dữ liệu và Ghi tệp...")
        embeddings_path = os.path.join(OUTPUT_DIR, "embeddings.npy")
        centrality_path = os.path.join(OUTPUT_DIR, "centrality.csv")
        
        np.save(embeddings_path, embeddings_matrix)
        centrality_df.to_csv(centrality_path)
        
        print(f"[HOÀN TẤT] File nén Node2Vec đã lưu: {embeddings_path}")
        print(f"[HOÀN TẤT] File chỉ số Node đã lưu: {centrality_path}")
        
    except Exception as e:
        print(f"Lỗi Hệ Thống: {e}")
