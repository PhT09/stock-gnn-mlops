"""
Parallel Correlation Computation Module
---------------------------------------
Mục tiêu: Tính toán ma trận tương quan Pearson đa luồng (song song trên nhiều lõi CPU)
từ dữ liệu Parquet lấy từ Spark, chuẩn bị cho Graph Engineering.
"""

# 1. IMPORTS
import os
import glob
import ast
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# 2. LOAD DATA
def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load dữ liệu chứng khoán từ thư mục Parquet.
    Kiểm tra các cột bắt buộc: date, ticker (symbol), và features.
    """
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Không tìm được file Parquet nào trong: {data_dir}")
        
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    
    # Validation cấu trúc cốt lõi
    required_cols = {'date', 'ticker', 'scaled_features'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dữ liệu thiếu cột bắt buộc. Cần có đủ: {required_cols}")
        
    return df

# 3. PREPROCESSING
def parse_spark_vector(vec):
    """Trích xuất mảng NumPy từ dict hoặc string do Spark serialize."""
    try:
        if isinstance(vec, dict) and 'values' in vec:
            return np.array(vec['values'])
        elif isinstance(vec, str):
            parsed_dict = ast.literal_eval(vec)
            return np.array(parsed_dict.get('values', []))
        return np.array([vec])
    except Exception:
        return np.array([])

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý làm sạch và Pivot bảng theo định dạng Timeline chuẩn.
    - Cột (Columns) = Mã cổ phiếu (symbol/ticker)
    - Dòng (Rows) = Ngày (date)
    - Giá trị (Values) = Proxy của Close Price
    """
    # 1. Giải phẫu Spark Vector lấy giá trị Đại diện (Close Price proxy)
    df['feature_vectors'] = df['scaled_features'].apply(parse_spark_vector)
    df['close'] = df['feature_vectors'].apply(lambda x: x[-1] if len(x) > 0 else np.nan)
    
    df = df.dropna(subset=['close'])
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Xoay trục theo Time-series
    pivot_df = df.pivot(index='date', columns='ticker', values='close')
    
    # 3. Handle missing values: Dồn trôi xuống (forward fill) ngày nghỉ lễ
    pivot_df = pivot_df.ffill().fillna(0)
    
    return pivot_df

# 4. PARALLEL CORRELATION FUNCTION
def _calc_pair_correlation(data_matrix: np.ndarray, i: int, j: int) -> float:
    """
    Hàm lõi (worker function) dùng cho Joblib multiprocessing.
    Tính hệ số Pearson giữa 2 vector cột i và j.
    """
    col_i = data_matrix[:, i]
    col_j = data_matrix[:, j]
    
    # Loại NaN riêng cho cặp nếu có (mặc dù đã fillna ở bước tiền xử lý)
    mask = ~np.isnan(col_i) & ~np.isnan(col_j)
    if np.sum(mask) < 2: 
        return 0.0 # Bỏ qua nếu data điểm chung quá bé
        
    # Tính ma trận 2x2 rồi lấy góc chéo
    return np.corrcoef(col_i[mask], col_j[mask])[0, 1]

def compute_correlation_parallel(pivot_df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Triển khai tính toán hệ số Tương quan Pearson sử dụng cơ chế Nhóm (Chunk/Multiprocessing).
    Cực kỳ tối ưu nhờ tận dụng khai thác tính Đối Xứng (Symmetry) của ma trận Correlation.
    """
    symbols = pivot_df.columns
    num_stocks = len(symbols)
    
    # Convert sang NumPy C-contiguous array để tối đa hoá tốc độ truyền cache cho CPU
    data_matrix = pivot_df.to_numpy(copy=True)
    
    # Khai thác tính ĐỐI XỨNG: Chỉ cần tính nửa trên của ma trận. Nửa dưới tự động copy qua.
    pairs = [(i, j) for i in range(num_stocks) for j in range(i + 1, num_stocks)]
    print(f"Đã phân rã bài toán thành {len(pairs)} node nhỏ. Phân giải trên {abs(n_jobs)} thread/core...")
    
    # Chạy phân tán song song qua CPU cores bằng Joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(_calc_pair_correlation)(data_matrix, i, j) for i, j in pairs
    )
    
    # Khởi tạo ma trận toàn zeros, và set sẵn Diagonal = 1.0
    corr_array = np.eye(num_stocks)
    
    # Lắp ráp ngươc kết quả từ CPU về ma trận NxN
    for idx, (i, j) in enumerate(pairs):
        corr_val = results[idx]
        # Xử lý an toàn NaN xuất hiện từ thuật toán chia 0 trong toán học
        if np.isnan(corr_val):
            corr_val = 0.0
            
        corr_array[i, j] = corr_val
        corr_array[j, i] = corr_val # Điền chéo đối xứng
        
    return pd.DataFrame(corr_array, index=symbols, columns=symbols)

# 5. EXECUTION + SAVE OUTPUT
if __name__ == "__main__":
    
    # Đường dẫn động bắt theo project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_IN_DIR = os.path.join(BASE_DIR, "downloaded_data")
    DATA_OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    
    try:
        # Mục 1: Đọc Dữ liệu
        print("Đang đọc luồng Mock Data...")
        df_raw = load_data(DATA_IN_DIR)
        
        # Mục 2: Biến đổi Time-series
        print(f"Đang Pivot Data cho {len(df_raw['ticker'].unique())} mã cổ phiếu...")
        df_pivot = preprocess_data(df_raw)
        
        # Mục 3: Multiprocessing Calculation
        print("Khởi động thuật toán Parallel Pearson Correlation...")
        # n_jobs=-1 = Dùng toàn bộ số Core của CPU máy
        correlation_matrix = compute_correlation_parallel(df_pivot, n_jobs=-1)
        
        # Mục 4: Xác thực kết quả xuất file
        output_csv_path = os.path.join(DATA_OUT_DIR, "parallel_correlation_matrix.csv")
        correlation_matrix.to_csv(output_csv_path)
        
        print("\n--- KẾT QUẢ ĐÃ XUẤT ---")
        print(f"Kích thước ma trận: {correlation_matrix.shape}")
        print(f"Lưu file CSV tại: {output_csv_path}")
        print("Trích xuất góc trên mẫu của Ma trận:")
        print(correlation_matrix.iloc[:3, :3])
        
    except Exception as e:
        print(f"Đã xảy ra lỗi thực thi: {e}")
