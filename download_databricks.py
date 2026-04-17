"""
Hướng dẫn cài đặt và thiết lập trước khi chạy script:

1. Cài đặt thư viện:
   Bạn cần cài đặt các thư viện cần thiết. Chạy lệnh sau trong terminal:
   pip install databricks-sdk python-dotenv

2. Cấu hình file .env:
   - Điền thông tin vào file `.env` vừa được tạo:
     DATABRICKS_HOST="https://dbc-d830d4f6-f7bb.cloud.databricks.com"
     DATABRICKS_TOKEN="chuỗi_token_của_bạn_vào_đây"

3. Chạy script:
   python download_databricks.py
"""

import os
import sys
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

def main():
    # Load biến môi trường từ file .env
    load_dotenv()

    # 1. Kiểm tra và lấy token/host từ file .env
    token = os.environ.get("DATABRICKS_TOKEN")
    host = os.environ.get("DATABRICKS_HOST")
    
    if not token or not host:
        print("Lỗi: Không tìm thấy DATABRICKS_TOKEN hoặc DATABRICKS_HOST trong file .env.")
        print("Vui lòng nhập đầy đủ thông tin vào file .env rồi chạy lại.")
        sys.exit(1)

    processed_path = "/Volumes/workspace/default/stock_data/processed/stock_features.parquet"
    # Thư mục trên máy tính để lưu trữ thư mục parquet tải về
    download_folder = "downloaded_data"

    try:
        print(f"Đang kết nối tới Databricks Workspace: {host}...")
        w = WorkspaceClient(host=host, token=token)

        # Spark thường lưu Parquet dưới dạng một THƯ MỤC chứa nhiều file part-0000...
        print(f"\nĐang kiểm tra thư mục Parquet trên Volume: {processed_path}")
        
        # 2. Liệt kê các file part trong thư mục parquet
        try:
            contents = w.files.list_directory_contents(processed_path)
            items = list(contents)
            print(f"Thư mục này chứa {len(items)} file con. Chuẩn bị tải toàn bộ thư mục...")
            
            # Tạo folder local nếu chưa có
            if not os.path.exists(download_folder):
                os.makedirs(download_folder)
            
            # 3. Tải tất cả các file bên trong thư mục
            for item in items:
                if not item.is_directory:
                    file_name = os.path.basename(item.path)
                    local_filepath = os.path.join(download_folder, file_name)
                    
                    print(f"Đang tải {file_name}...")
                    response = w.files.download(item.path)
                    
                    with open(local_filepath, 'wb') as f:
                        f.write(response.contents.read())
            
            print(f"\nThành công! Toàn bộ thư mục Parquet đã được lưu xuống folder '{download_folder}'.")
            
        except Exception as dir_e:
            if "not exist" in str(dir_e).lower() or "404" in str(dir_e):
                print(f"Lỗi: Thư mục {processed_path} hoàn toàn không tồn tại.")
            else:
                raise dir_e

    except Exception as e:
        # 4. Best practices: Xử lý ngoại lệ rõ ràng khi xảy ra vấn đề
        print("\n=== ĐÃ XẢY RA LỖI KHI XỬ LÝ ===")
        print(f"Chi tiết lỗi: {str(e)}")
        print("\nVui lòng kiểm tra lại các nguyên nhân sau:")
        print(" 1. Token của bạn có chính xác và còn quyền truy cập không?")
        print(" 2. Đường dẫn Volume kia có tồn tại trên Workspace không?")
        print(" 3. Kiểm tra kết nối Internet của bạn.")

if __name__ == "__main__":
    main()
