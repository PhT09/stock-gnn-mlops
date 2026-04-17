"""
Graph Construction Module cho MLOps Pipeline
-------------------------------------------
Mục tiêu: Đọc dữ liệu chứng khoán dạng Parquet, xử lý giá trị khuyết, xoay trục thời gian (Time-series),
tính ma trận tương quan Pearson và dựng Đồ thị (Graph) vô hướng bằng NetworkX dựa trên độ tương đồng (Threshold).
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import glob
import ast
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# =====================================================================
# 2. LOAD DATA
# =====================================================================
def load_mock_data(data_dir: str) -> pd.DataFrame:
    """
    Đọc tất cả các file Parquet chứa dữ liệu chứng khoán trong chỉ định.
    Args:
        data_dir (str): Thư mục chứa các file .parquet.
    Returns:
        pd.DataFrame: Bảng DataFrame tổng hợp chứa toàn bộ dữ liệu.
    """
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy file Parquet nào trong thư mục: {data_dir}")
        
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    return df

# =====================================================================
# 3. PREPROCESSING
# =====================================================================
def parse_spark_vector(vec):
    """Trích xuất mảng giá trị (NumPy array) từ cột Spark MLlib Vector."""
    try:
        if isinstance(vec, dict) and 'values' in vec:
            return np.array(vec['values'])
        elif isinstance(vec, str):
            parsed_dict = ast.literal_eval(vec)
            return np.array(parsed_dict.get('values', []))
        return np.array([vec])
    except Exception:
        return np.array([])

def preprocess_and_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý dữ liệu đặc trưng và Pivot mảng dữ liệu.
    - Cột (Columns) = Stock Symbols (ticker)
    - Dòng (Rows) = Dates
    - Giá trị (Values) = Giá đóng cửa hoặc thuộc tính đặc trưng đại diện
    - Trám giá trị rỗng (Handle missing values) bằng Forward Fill.
    """
    # Trích xuất dữ liệu mảng và chọn phần tử cuối cùng làm Giá Trị Biến Động (Closing Proxy)
    df['feature_vectors'] = df['scaled_features'].apply(parse_spark_vector)
    df['close_price_proxy'] = df['feature_vectors'].apply(lambda x: x[-1] if len(x) > 0 else np.nan)
    
    # Định dạng mốc thời gian và làm sạch dữ liệu nhiễu
    df = df.dropna(subset=['close_price_proxy'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'ticker'])
    
    # Pivot sang định dạng [N Ngày x M Chứng khoán]
    pivot_df = df.pivot(index='date', columns='ticker', values='close_price_proxy')
    
    # Kỹ thuật Fill Missing Data: Kéo giá ngày hôm trước xuống gán cho ngày lễ/cuối tuần
    pivot_df = pivot_df.ffill().fillna(0)
    
    return pivot_df

# =====================================================================
# 4. CORRELATION CALCULATION
# =====================================================================
def calculate_correlation(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán ma trận tương quan Pearson."""
    # Hàm .corr() tự bỏ qua NaN và sinh ra ma trận N x N
    corr_matrix = pivot_df.corr(method='pearson')
    
    # Ở Pandas >= 2.0, .values là read-only. Phải clone sang numpy array để tránh lỗi
    corr_values = corr_matrix.to_numpy(copy=True)
    np.fill_diagonal(corr_values, 0.0)
    
    return pd.DataFrame(corr_values, index=corr_matrix.index, columns=corr_matrix.columns)

# =====================================================================
# 5. GRAPH CONSTRUCTION
# =====================================================================
def build_networkx_graph(corr_matrix: pd.DataFrame, threshold: float = 0.6) -> nx.Graph:
    """
    Dựng Đồ thị Vô hướng bằng NetworkX. Nối Cạnh giữa hai đỉnh nếu Tương quan > Threshold.
    """
    G = nx.Graph()
    tickers = corr_matrix.columns.tolist()
    
    # Nạp toàn bộ các đỉnh (Nodes) trước khi kết nối (tránh mất đỉnh khi đỉnh không có edges)
    G.add_nodes_from(tickers)
    
    # Nối Cạnh (Edges) từ ma trận
    num_tickers = corr_matrix.shape[0]
    
    for i in range(num_tickers):
        for j in range(i + 1, num_tickers): # Quét nửa trên tam giác (Upper Triangle)
            stock_a = tickers[i]
            stock_b = tickers[j]
            corr = corr_matrix.iloc[i, j]
            
            # Chỉ nối khi đảm bảo độ tin cậy và thỏa ngưỡng Threshold
            if pd.notna(corr) and abs(corr) > threshold:
                G.add_edge(stock_a, stock_b, weight=round(corr, 4))
                
    return G

# =====================================================================
# 6. EXPORT RESULTS
# =====================================================================
def export_edges_to_csv(G: nx.Graph, output_dir: str = "output", filename: str = "edge_list.csv"):
    """Lấy danh sách Edges từ Graph và xuất ra file CSV dùng cho ML Training."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    
    edges_df = nx.to_pandas_edgelist(G)
    
    if edges_df.empty:
        edges_df = pd.DataFrame(columns=["source", "target", "weight"])
        print("Lưu ý: Không tìm được sự tương quan thỏa điều kiện Threshold.")
        
    edges_df.to_csv(out_path, index=False)
    print(f"Đã xuất dữ liệu cạnh (Edges) tại: {out_path}")

# =====================================================================
# 7. (OPTIONAL) VISUALIZATION
# =====================================================================
def visualize_graph(G: nx.Graph):
    """Vẽ mạng lưới đồ thị trực quan cho mẫu vừa tính."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42) # Layout phân bổ tự nhiên
    
    # Trích xuất thuộc tính weight (lực tương tác)
    edges = G.edges(data=True)
    weights = [attr['weight'] for u, v, attr in edges] if edges else []
    
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    plt.title("Stock Market Correlation Graph")
    plt.axis('off')
    
    # Lưu ra file thay vì show để tránh lỗi môi trường không có UI (như Github Actions/Databricks)
    plt.savefig("graph_visualization.png")
    print("Đã vẽ xong và lưu thành công cấu trúc mạng lưới ra file: graph_visualization.png")

# =====================================================================
# CHẠY TỔNG HỢP (MAIN ENGINE)
# =====================================================================
if __name__ == "__main__":
    
    # Do file này đang được chạy từ thư mục gốc của project (c:\...\stock-gnn-mlops)
    # Đường dẫn đến file data sẽ là:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIRECTORY = os.path.join(base_dir, "downloaded_data")
    
    CORRELATION_THRESHOLD = 0.6  # Có thể tùy chỉnh (configurable)
    
    try:
        # Bước 2: Load dữ liệu
        print("\n=== KHỞI ĐỘNG GRAPH CONSTRUCTION MODULE ===")
        print("Đang khởi chạy luồng Đọc dữ liệu...")
        raw_df = load_mock_data(DATA_DIRECTORY)
        
        # Bước 3: Tiền Xử lý
        print("Đang cấu trúc lại chuỗi thời gian (Pivot)...")
        pivot_data = preprocess_and_pivot(raw_df)
        
        # Bước 4: Tính Ma trận Tương Quan
        print("Đang tính Ma trận Pearson Correlation...")
        correlation_df = calculate_correlation(pivot_data)
        
        # Bước 5: Build Đồ thị
        print(f"Đang dựng NetworkX Graph (ngưỡng cắt={CORRELATION_THRESHOLD})...")
        stock_graph = build_networkx_graph(correlation_df, threshold=CORRELATION_THRESHOLD)
        
        print("\n--- BÁO CÁO THỐNG KÊ ĐỒ THỊ ---")
        print(f"Số lượng Đỉnh (Nodes): {stock_graph.number_of_nodes()}")
        print(f"Số lượng Cạnh (Edges): {stock_graph.number_of_edges()}")
        
        # Bước 6: Xuất kết quả csv
        output_folder = os.path.join(base_dir, "data", "graph")
        export_edges_to_csv(stock_graph, output_dir=output_folder, filename="mock_edges.csv")
        
        # Bước 7 (Optional): Gọi nếu muốn xem trực quan
        visualize_graph(stock_graph)
        
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
