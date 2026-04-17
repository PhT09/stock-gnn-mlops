import pandas as pd
import os

def analyze_stock_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy file {file_path}")
        return

    try:
        # 1. Đọc file
        df = pd.read_csv(file_path)
        
        print("\n" + "="*50)
        print("📊 PHÂN TÍCH LUỒNG DỮ LIỆU CỔ PHIẾU (ROLE D)")
        print("="*50)
        
        # 2. Xem 5 dòng đầu
        print("\n🔍 5 Dòng dữ liệu đầu tiên:")
        print(df.head())
        
        # 3. Chuẩn hóa ngày tháng
        df['date'] = pd.to_datetime(df['date'])
        
        # 4. Kiểm tra kiểu dữ liệu và giá trị thiếu
        print("\n📝 Thông tin cấu trúc dữ liệu:")
        print(df.info())
        
        # 5. Thống kê theo mã cổ phiếu (Ticker)
        print("\n📈 Số lượng bản ghi theo mã cổ phiếu (Ticker):")
        ticker_counts = df['ticket'].value_counts()
        for ticket, count in ticker_counts.items():
            print(f"   - {ticket}: {count} phiên giao dịch")
        
        # 6. Tóm tắt thống kê số học
        print("\n⚖️ Tóm tắt thống kê các chỉ số (OHLCV):")
        print(df.describe())

        print("\n✅ KIỂM TRA HOÀN TẤT: Luồng dữ liệu sẵn sàng cho xử lý GNN.")
        print("="*50 + "\n")

    except Exception as e:
        print(f"⚠️ Đã xảy ra lỗi khi phân tích: {e}")

if __name__ == "__main__":
    # Đường dẫn file dữ liệu mẫu
    DATA_FILE = 'mock_data.csv'
    analyze_stock_data(DATA_FILE)
