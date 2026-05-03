import pandas as pd

def show_data_sample(data_path="data/raw/stock_data"):
    try:
        # Đọc dữ liệu từ thư mục parquet
        df = pd.read_parquet(data_path)
        
        print("\n" + "="*50)
        print("📊 TỔNG QUAN DỮ LIỆU (DATA OVERVIEW)")
        print("="*50)
        print(f"Tổng số dòng: {len(df):,}")
        print(f"Các cột dữ liệu: {df.columns.tolist()}")
        
        print("\n" + "="*50)
        print("👀 5 DÒNG ĐẦU TIÊN (FIRST 5 ROWS)")
        print("="*50)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.head(5))
        
        print("\n" + "="*50)
        print("🔬 CHI TIẾT CỘT SCALED_FEATURES (Spark DenseVector)")
        print("="*50)
        # Xem thử dòng đầu tiên của cột feature nén
        first_feature = df['scaled_features'].iloc[0]
        print(f"Loại dữ liệu: {type(first_feature)}")
        if isinstance(first_feature, dict) and 'values' in first_feature:
            print(f"Các chỉ số (features) bên trong: \n{first_feature['values']}")
            print(f"Tổng cộng có {len(first_feature['values'])} biến features (như MA, RSI, MACD...) đã được gom lại.")
            
    except Exception as e:
        print(f"Không thể đọc được dữ liệu: {e}")
        print("Đảm bảo bạn đã chạy xong quá trình tải data về thư mục data/raw/stock_data")

if __name__ == "__main__":
    show_data_sample()
