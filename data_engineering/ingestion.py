import os
import sys
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

# Thêm Databricks CLI vào PATH để Python SDK có thể tự động lấy token OAuth từ file .exe
cli_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "databricks_cli_folder")
os.environ["PATH"] = cli_path + os.pathsep + os.environ.get("PATH", "")

def ingest_data(output_folder="data/raw/stock_data"):
    """
    Ingest data by downloading it from Databricks Volume.
    The Databricks jobs already scrape the data daily and save it as parquet.
    """
    print("--- Bắt đầu tiến trình Ingestion từ Databricks ---")
    load_dotenv()

    host = os.environ.get("DATABRICKS_HOST")
    
    if not host:
        print("Lỗi: Không tìm thấy DATABRICKS_HOST trong file .env.")
        return False

    processed_path = "/Volumes/workspace/default/stock_data/processed/stock_features.parquet"

    try:
        print(f"Đang kết nối tới Databricks bằng Profile: dbc-d830d4f6-f7bb...")
        # Sử dụng chính xác tên profile mà Databricks CLI vừa tạo
        w = WorkspaceClient(profile="dbc-d830d4f6-f7bb")
        
        print(f"Đang kiểm tra thư mục Parquet trên Volume: {processed_path}")
        
        try:
            contents = w.files.list_directory_contents(processed_path)
            items = list(contents)
            
            os.makedirs(output_folder, exist_ok=True)
            
            # Lấy danh sách tên file trên Databricks
            remote_filenames = [os.path.basename(item.path) for item in items if not item.is_directory]
            
            # 1. Xóa các file cũ ở local không còn tồn tại trên Databricks (tránh nhân đôi dữ liệu)
            local_files = os.listdir(output_folder)
            for lf in local_files:
                if lf not in remote_filenames:
                    os.remove(os.path.join(output_folder, lf))
            
            # 2. Chỉ tải về những file chưa có ở local
            download_count = 0
            for item in items:
                if not item.is_directory:
                    file_name = os.path.basename(item.path)
                    local_filepath = os.path.join(output_folder, file_name)
                    
                    if file_name not in local_files:
                        print(f"Đang tải file mới: {file_name}...")
                        response = w.files.download(item.path)
                        with open(local_filepath, 'wb') as f:
                            f.write(response.contents.read())
                        download_count += 1
            
            print(f"Thành công! Đã đồng bộ {download_count} file mới. Dữ liệu nằm ở: '{output_folder}'.")
            return True
            
        except Exception as dir_e:
            if "not exist" in str(dir_e).lower() or "404" in str(dir_e):
                print(f"Lỗi: Thư mục {processed_path} hoàn toàn không tồn tại trên Databricks.")
            else:
                raise dir_e

    except Exception as e:
        print(f"Lỗi trong quá trình Ingestion: {str(e)}")
        return False

if __name__ == "__main__":
    ingest_data()
