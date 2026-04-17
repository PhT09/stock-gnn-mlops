import glob
import pandas as pd
import numpy as np
import os
import networkx as nx

def parse_spark_vector(vec):
    """
    Hàm này dùng để trích xuất mảng giá trị (NumPy array)
    từ cột scaled_features do Spark MLlib sinh ra (dict).
    """
    try:
        if isinstance(vec, dict) and 'values' in vec:
            return np.array(vec['values'])
        elif isinstance(vec, str):
            import ast
            parsed_dict = ast.literal_eval(vec)
            return np.array(parsed_dict.get('values', []))
        return np.array([vec])
    except Exception as e:
        return np.array([])

def step1_load_and_preprocess():
    print("--- BƯỚC 1: LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU ---")
    
    # Đảm bảo đường dẫn chính xác dù chạy từ root hay từ trong graph_module
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "downloaded_data")
    
    # 1. Tìm tất cả các file parquet
    parquet_files = glob.glob(f"{data_path}/*.parquet")
    if not parquet_files:
        print(f"Lỗi: Không tìm thấy file Parquet nào trong thư mục {data_path}")
        return None
        
    print(f"Đã tìm thấy {len(parquet_files)} file Parquet. Đang tiến hành đọc...")
    
    # 2. Đọc và ghép nối (concatenate) toàn bộ lại thành 1 DataFrame
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    
    # 3. Trích xuất mảng `values` từ cấu trúc Spark Vector
    print("Đang bóc tách cột scaled_features (Trích xuất các Vector đặc trưng)...")
    df['features_array'] = df['scaled_features'].apply(parse_spark_vector)
    
    # 4. Tính ma trận biến động: Lấy ví dụ giá trị cuối của vector làm đại diện (giả định là Close price/Return)
    df['representative_value'] = df['features_array'].apply(lambda x: x[-1] if len(x) > 0 else 0)
    
    # 5. Làm sạch dữ liệu
    df = df.dropna(subset=['representative_value'])
    df['date'] = pd.to_datetime(df['date'])  # Đảm bảo định dạng thời gian chuẩn tốc độ cao
    df = df.sort_values(by=['date', 'ticker'])
    
    # 6. TỐI ƯU HÓA: Tạo Pivot Table (Ma Trận Time-Series)
    # Cột là Mã cổ phiếu, Dòng là Ngày, Giá trị là Biến động (để tính toán Mạng Lưới cực nhanh ở Bước 2)
    pivot_df = df.pivot(index='date', columns='ticker', values='representative_value')
    
    # Trám dữ liệu thiếu (nội suy) nếu cổ phiếu có ngày không giao dịch
    pivot_df = pivot_df.ffill().fillna(0)
    
    print(f"[THÀNH CÔNG BƯỚC 1] Tổng số ngày: {pivot_df.shape[0]} | Tổng số mã cổ phiếu (Nodes): {pivot_df.shape[1]}")
    return pivot_df

def step2_build_graph(pivot_df, threshold=0.65):
    print("\n--- BƯỚC 2: XÂY DỰNG ĐỒ THỊ BẰNG NETWORKX ---")
    
    # 1. Tính toán ma trận tương quan Pearson giữa tất cả các cột(cổ phiếu) với nhau
    print("Đang tính toán Ma trận Tương quan (Pearson Correlation)...")
    corr_matrix = pivot_df.corr(method='pearson')
    
    # 2. Khởi tạo một đồ thị vô hướng
    G = nx.Graph()
    
    # KHIẾM KHUYẾT TRƯỚC ĐÓ: Nếu 2 mã không có dòng liên kết nào thì nó sẽ bị rớt khỏi Graph.
    # THẾ NÊN PHẢI: Khai báo toàn bộ Đỉnh (Nodes) trước khi xét Cạnh (Edges)
    tickers = corr_matrix.columns
    G.add_nodes_from(tickers)
    
    # 3. Quét qua Ma trận Tương quan để nối Cạnh (Edge)
    num_tickers = corr_matrix.shape[0]
    edge_count = 0
    
    for i in range(num_tickers):
        for j in range(i + 1, num_tickers): # Chỉ quét nửa trên ma trận (tránh lặp A-B và B-A)
            stock_A = tickers[i]
            stock_B = tickers[j]
            correlation_score = corr_matrix.iloc[i, j]
            
            # 4. Nếu hệ số tương quan vượt Threshold, nối chúng lại!
            # Tránh lỗi NaN (chia cho 0 nếu giá trị tĩnh)
            if pd.notna(correlation_score) and abs(correlation_score) >= threshold:
                G.add_edge(stock_A, stock_B, weight=correlation_score)
                edge_count += 1
                
    print(f"[THÀNH CÔNG BƯỚC 2] Đã dựng xong Đồ thị Stock Graph!")
    print(f"Tổng số Đỉnh (Nodes): {G.number_of_nodes()}")
    print(f"Tổng số Cạnh (Edges): {G.number_of_edges()} (với Threshold = {threshold})")
    
    return G

def step3_export_graph_data(G, output_csv="graph_edges.csv"):
    print("\n--- BƯỚC 3: XUẤT DỮ LIỆU ĐỒ THỊ CHO MACHINE LEARNING ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data", "graph")
    os.makedirs(output_dir, exist_ok=True) # Đảm bảo thư mục tồn tại
    
    output_path = os.path.join(output_dir, output_csv)
    
    # Rút trích danh sách Cạnh (Edges) từ đồ thị NetworkX
    edges = nx.to_pandas_edgelist(G)
    
    # Kiểm tra xem đồ thị có cạnh không
    if edges.empty:
        # Nếu chưa có cạnh, tạo bảng rỗng với đúng chuẩn cột cho chuẩn cấu trúc
        edges = pd.DataFrame(columns=["source", "target", "weight"])
        print("Lưu ý: Đồ thị hiện đang không có cạnh nào. Một file cấu trúc rỗng đã được tạo.")
    else:
        # Đổi tên cho đúng chuẩn file dataset.py của team C yêu cầu
        edges = edges.rename(columns={"source": "source", "target": "target"})
        if "weight" not in edges.columns:
            edges["weight"] = 1.0 # Default weight
            
    # Lưu ra định dạng CSV không lấy cột index
    edges.to_csv(output_path, index=False)
    
    print(f"[THÀNH CÔNG BƯỚC 3] Đã xuất file mô tả Đồ thị ra: {output_path}")
    print(edges.head(5))

def step4_generate_embeddings(G, output_npy="stock_embeddings.npy"):
    print("\n--- BƯỚC 4: TẠO EMBEDDING VỚI NODE2VEC ---")
    
    from node2vec import Node2Vec
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "data", "graph")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_npy)
    
    nodes = list(G.nodes())
    if len(nodes) == 0:
        print("Đồ thị không có đỉnh nào, bỏ qua Bước 4.")
        return
        
    if G.number_of_edges() == 0:
        print("Lưu ý: Đồ thị chưa có cạnh nối. Sinh Embedding độc lập cho mỗi Đỉnh (Zero Matrix)...")
        # Khởi tạo chuỗi bằng 0 cho các Đỉnh vì Node2Vec yêu cầu có đường đi (walk)
        # Chiều Dimentions thường set = 32 hoặc 64 (giữ 32 cho đồng dạng thuật toán GNN phía sau)
        embedding_matrix = np.zeros((len(nodes), 32))
        np.save(output_path, embedding_matrix)
        print(f"[THÀNH CÔNG BƯỚC 4] Đã lưu mô hình Zero Tensor ra: {output_path}")
        return
        
    print(f"Đang phân tích Walks và tính toán Node2Vec cho {len(nodes)} Đỉnh...")
    
    try:
        # P và Q là các hệ số khám phá cục bộ và toàn cục
        node2vec = Node2Vec(G, dimensions=32, walk_length=15, num_walks=50, workers=1, quiet=True)
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        
        # Sắp xếp các Embeddings y hệt thứ tự của `G.nodes()`
        # Team C (Machine learning) của bạn sẽ đọc Array này theo Index của G.nodes()
        embedding_matrix = np.array([model.wv[node] for node in nodes])
        
        # Lưu ra File NPY
        np.save(output_path, embedding_matrix)
        print(f"[THÀNH CÔNG BƯỚC 4] Khai phá xong! File mảng 2 chiều mô tả Đỉnh đã lưu ở: {output_path}")
        print(f"Kích thước Matrix: {embedding_matrix.shape}")
        
    except Exception as e:
        print(f"Lỗi khi chạy Node2Vec: {e}")

if __name__ == "__main__":
    # ===== CHẠY QUY TRÌNH TOÀN BỘ =====
    # Khởi động Bước 1
    df_pivot = step1_load_and_preprocess()
    
    # Khởi động Bước 2
    if df_pivot is not None:
        stock_graph = step2_build_graph(df_pivot, threshold=0.65)
        
        # Khởi động Bước 3
        step3_export_graph_data(stock_graph, output_csv="stock_edges.csv")
        
        # Khởi động Bước 4
        step4_generate_embeddings(stock_graph, output_npy="stock_embeddings.npy")
