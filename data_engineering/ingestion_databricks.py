import os
import json
from datetime import datetime, timedelta
import pandas as pd

CACHE_FILE = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/data/.last_processed_date.json"

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

def ingest_data_databricks(output_folder="data/raw/stock_data", recent_days=None, check_freshness=True, force=False):
    """
    Đọc data trực tiếp từ Databricks Volume (cho Databricks Job/Notebook).
    
    Args:
        output_folder (str): Folder để lưu parquet files (local trong Workspace)
        recent_days (int, optional): Nếu set, chỉ giữ N ngày gần nhất
        check_freshness (bool): Cảnh báo nếu data cũ > 2 ngày
        force (bool): Bắt buộc xử lý dù data không đổi
        
    Returns:
        dict: Metadata về data (has_new_data, latest_date, etc.)
    """
    print("--- Bắt đầu tiến trình Ingestion từ Databricks Volume ---")
    
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        
        volume_path = "/Volumes/workspace/default/stock_data/processed/stock_features.parquet"
        
        print(f"Đang đọc data từ: {volume_path}")
        
        # Đọc từ Volume bằng Spark
        df_spark = spark.read.parquet(volume_path)
        
        # Get metadata
        row_count = df_spark.count()
        latest_date = df_spark.agg({"date": "max"}).collect()[0][0]
        oldest_date = df_spark.agg({"date": "min"}).collect()[0][0]
        unique_dates = df_spark.select("date").distinct().count()
        
        latest_date_str = str(latest_date.date())
        oldest_date_str = str(oldest_date.date())
        
        print(f"\n✅ Data Summary:")
        print(f"   From: {oldest_date_str} to {latest_date_str}")
        print(f"   Total rows: {row_count:,}")
        print(f"   Unique dates: {unique_dates}")
        
        # Check for new data
        last_processed_date, last_processed_at = load_last_processed_date()
        
        has_new_data = False
        if last_processed_date is None:
            has_new_data = True
            print(f"\n🆕 ĐÂY LÀ LẦN ĐẦU TIÊN ingest data")
        elif latest_date_str > last_processed_date:
            has_new_data = True
            print(f"\n🆕 CÓ DATA MỚI!")
            print(f"   Lần trước: {last_processed_date} (xử lý lúc {last_processed_at})")
            print(f"   Bây giờ:   {latest_date_str}")
        else:
            print(f"\n⏸️  KHÔNG CÓ DATA MỚI")
            print(f"   Latest date: {latest_date_str} (giống lần trước: {last_processed_date})")
            if not force:
                print(f"   → Bỏ qua xử lý. Dùng force=True để bắt buộc chạy.")
        
        # Check data freshness
        warnings = []
        days_old = (datetime.now() - latest_date).days
        if check_freshness and days_old > 2:
            warning_msg = f"⚠️ CẢNH BÁO: Dữ liệu đã cũ {days_old} ngày (mới nhất: {latest_date_str})"
            print(warning_msg)
            warnings.append(warning_msg)
        
        # Convert to Pandas and save locally (for training)
        print(f"\nĐang convert sang Pandas và lưu local...")
        os.makedirs(output_folder, exist_ok=True)
        
        if recent_days:
            print(f"Lọc chỉ lấy {recent_days} ngày gần nhất...")
            cutoff_date = latest_date - timedelta(days=recent_days)
            df_spark_filtered = df_spark.filter(df_spark.date >= cutoff_date)
            df_pandas = df_spark_filtered.toPandas()
            print(f"✅ Đã lọc: {len(df_pandas):,} dòng")
        else:
            df_pandas = df_spark.toPandas()
        
        # Save as parquet
        df_pandas.to_parquet(output_folder, index=False)
        print(f"✅ Đã lưu tại: {output_folder}")
        
        # Update cache
        if has_new_data or force:
            save_last_processed_date(latest_date_str)
            print(f"\n💾 Đã lưu cache: {latest_date_str}")
        
        metadata = {
            "success": True,
            "latest_date": latest_date_str,
            "oldest_date": oldest_date_str,
            "total_rows": row_count,
            "unique_dates": unique_dates,
            "days_old": days_old,
            "warnings": warnings,
            "output_folder": output_folder,
            "has_new_data": has_new_data,
            "last_processed_date": last_processed_date,
            "last_processed_at": last_processed_at
        }
        
        print(f"\n✅ Thành công!")
        return metadata
        
    except Exception as e:
        error_msg = f"Lỗi: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": error_msg}

# Wrapper để tự động detect environment
def ingest_data(output_folder="data/raw/stock_data", recent_days=None, check_freshness=True, force=False):
    """
    Auto-detect environment và chọn phương thức ingestion phù hợp.
    """
    # Kiểm tra có đang chạy trong Databricks không
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        print("🔍 Detected: Databricks environment")
        return ingest_data_databricks(output_folder, recent_days, check_freshness, force)
    else:
        print("🔍 Detected: Local environment")
        # Import local version (with WorkspaceClient)
        from databricks.sdk import WorkspaceClient
        from dotenv import load_dotenv
        load_dotenv()
        # ... (code download version cũ)
        # Để đơn giản, tạm thời raise error
        raise NotImplementedError("Local ingestion - use original ingestion.py")

if __name__ == "__main__":
    metadata = ingest_data()
    print(f"\n📊 METADATA:")
    import json
    print(json.dumps(metadata, indent=2, default=str))
