import os
import sys
import pandas as pd
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

# Thêm Databricks CLI vào PATH
cli_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "databricks_cli_folder")
os.environ["PATH"] = cli_path + os.pathsep + os.environ.get("PATH", "")

CACHE_FILE = "data/.last_processed_date.json"

def load_last_processed_date():
    """Load the last processed date from cache"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                return cache.get('latest_date'), cache.get('processed_at')
        except:
            pass
    return None, None

def save_last_processed_date(latest_date):
    """Save the latest processed date to cache"""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    cache = {
        'latest_date': latest_date,
        'processed_at': datetime.now().isoformat()
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def ingest_data(output_folder="data/raw/stock_data", recent_days=None, check_freshness=True, force=False):
    """
    Ingest data by downloading it from Databricks Volume.
    
    Args:
        output_folder (str): Local folder to save downloaded data
        recent_days (int, optional): If set, only keep data from last N days
        check_freshness (bool): If True, warn if data is older than 2 days
        force (bool): If True, process even if data hasn't changed
        
    Returns:
        dict: Metadata including has_new_data flag
    """
    print("--- Bắt đầu tiến trình Ingestion từ Databricks ---")
    load_dotenv()

    host = os.environ.get("DATABRICKS_HOST")
    
    if not host:
        print("Lỗi: Không tìm thấy DATABRICKS_HOST trong file .env.")
        return {"success": False, "error": "Missing DATABRICKS_HOST"}

    processed_path = "/Volumes/workspace/default/stock_data/processed/stock_features.parquet"

    try:
        print(f"Đang kết nối tới Databricks...")
        w = WorkspaceClient()
        
        print(f"Đang kiểm tra thư mục Parquet trên Volume: {processed_path}")
        
        try:
            contents = w.files.list_directory_contents(processed_path)
            items = list(contents)
            
            os.makedirs(output_folder, exist_ok=True)
            
            # Lấy danh sách tên file trên Databricks
            remote_filenames = [os.path.basename(item.path) for item in items if not item.is_directory]
            
            # 1. Xóa các file cũ ở local không còn tồn tại trên Databricks
            local_files = os.listdir(output_folder) if os.path.exists(output_folder) else []
            for lf in local_files:
                if lf not in remote_filenames:
                    os.remove(os.path.join(output_folder, lf))
                    print(f"Đã xóa file cũ: {lf}")
            
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
            
            print(f"Đã đồng bộ {download_count} file mới.")
            
            # 3. Đọc data để phân tích metadata
            print(f"\nĐang phân tích metadata...")
            df = pd.read_parquet(output_folder)
            
            # Phát hiện latest date
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                latest_date = df['date'].max()
                oldest_date = df['date'].min()
                unique_dates = df['date'].nunique()
                total_rows = len(df)
                
                # 4. CHECK FOR NEW DATA (So sánh với lần trước)
                last_processed_date, last_processed_at = load_last_processed_date()
                
                has_new_data = False
                if last_processed_date is None:
                    has_new_data = True
                    print(f"🆕 ĐÂY LÀ LẦN ĐẦU TIÊN ingest data")
                elif str(latest_date.date()) > last_processed_date:
                    has_new_data = True
                    print(f"🆕 CÓ DATA MỚI!")
                    print(f"   Lần trước: {last_processed_date} (xử lý lúc {last_processed_at})")
                    print(f"   Bây giờ:   {latest_date.date()}")
                else:
                    print(f"⏸️  KHÔNG CÓ DATA MỚI")
                    print(f"   Latest date: {latest_date.date()} (giống lần trước: {last_processed_date})")
                    if not force:
                        print(f"   → Bỏ qua xử lý. Dùng force=True để bắt buộc chạy.")
                
                print(f"\n✅ Dữ liệu từ {oldest_date.date()} đến {latest_date.date()}")
                print(f"✅ Tổng {unique_dates} ngày giao dịch, {total_rows:,} dòng dữ liệu")
                
                # 5. Check data freshness
                warnings = []
                days_old = (datetime.now() - latest_date).days
                if check_freshness and days_old > 2:
                    warning_msg = f"⚠️ CẢNH BÁO: Dữ liệu đã cũ {days_old} ngày (mới nhất: {latest_date.date()})"
                    print(warning_msg)
                    warnings.append(warning_msg)
                
                # 6. Optional: Filter to recent days only
                if recent_days is not None:
                    cutoff_date = latest_date - timedelta(days=recent_days)
                    df_filtered = df[df['date'] >= cutoff_date]
                    filtered_path = output_folder.replace('/stock_data', '/stock_data_filtered')
                    os.makedirs(filtered_path, exist_ok=True)
                    df_filtered.to_parquet(filtered_path, index=False)
                    print(f"✅ Đã lọc {len(df_filtered):,} dòng từ {recent_days} ngày gần nhất")
                    print(f"✅ Lưu tại: {filtered_path}")
                
                # 7. Save to cache (only if has new data or force)
                if has_new_data or force:
                    save_last_processed_date(str(latest_date.date()))
                    print(f"\n💾 Đã lưu cache: {latest_date.date()}")
                
                metadata = {
                    "success": True,
                    "latest_date": str(latest_date.date()),
                    "oldest_date": str(oldest_date.date()),
                    "total_rows": total_rows,
                    "unique_dates": unique_dates,
                    "days_old": days_old,
                    "warnings": warnings,
                    "output_folder": output_folder,
                    "has_new_data": has_new_data,
                    "last_processed_date": last_processed_date,
                    "last_processed_at": last_processed_at
                }
                
                print(f"\n✅ Thành công! Dữ liệu nằm ở: '{output_folder}'")
                return metadata
            else:
                print("⚠️ Không tìm thấy cột 'date' trong dữ liệu")
                return {"success": True, "warning": "No date column found"}
            
        except Exception as dir_e:
            if "not exist" in str(dir_e).lower() or "404" in str(dir_e):
                error_msg = f"Lỗi: Thư mục {processed_path} không tồn tại trên Databricks."
                print(error_msg)
                return {"success": False, "error": error_msg}
            else:
                raise dir_e

    except Exception as e:
        error_msg = f"Lỗi trong quá trình Ingestion: {str(e)}"
        print(error_msg)
        return {"success": False, "error": error_msg}

if __name__ == "__main__":
    # Test
    metadata = ingest_data()
    print(f"\n📊 METADATA: {metadata}")
