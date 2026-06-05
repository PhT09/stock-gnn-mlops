import os
import json
from datetime import datetime
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from data_engineering.preprocessing import preprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def ingest_data(output_folder="/Volumes/workspace/default/stock_data/processed/stock_features.parquet",
                recent_days=None,
                check_freshness=True,
                force=False):
    """
    Ingest data on Databricks:
    - Reads raw parquet from Volumes (or local fallback)
    - Checks cache to determine if there is new data
    - Runs preprocessing logic and saves to output_folder
    - Updates cache and returns execution metadata
    """
    logger.info("=" * 80)
    logger.info("DATABRICKS INGESTION & CACHE CHECK")
    logger.info("=" * 80)
    
    # 1. Determine raw path
    raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    if not os.path.exists(raw_path) and "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        local_raw = "downloaded_data"
        if os.path.exists(local_raw):
            logger.info(f"Local environment: falling back raw_path to {local_raw}")
            raw_path = local_raw
            
    # If raw path still doesn't exist, return error
    if not os.path.exists(raw_path):
        return {
            "success": False,
            "error": f"Raw data parquet path not found: {raw_path}"
        }
        
    try:
        # 2. Initialize Spark
        spark = SparkSession.builder.appName("DatabricksIngestion").getOrCreate()
        
        # 3. Read raw data to check latest date
        df_raw = spark.read.parquet(raw_path)
        latest_date_val = df_raw.selectExpr("max(date) as max_date").collect()[0][0]
        
        if not latest_date_val:
            return {
                "success": False,
                "error": "No dates found in raw stock data"
            }
            
        latest_date_str = str(latest_date_val)
        logger.info(f"Latest date in raw data: {latest_date_str}")
        
        # Parse date to calculate days_old
        clean_date_str = latest_date_str.split('(')[0]
        try:
            latest_date_dt = datetime.strptime(clean_date_str, '%Y-%m-%d')
            days_old = (datetime.now() - latest_date_dt).days
        except Exception as ex:
            logger.warning(f"Could not parse date {clean_date_str}: {ex}")
            days_old = 0
            
        # 4. Cache check logic
        # Cache paths: Databricks vs Local
        workspace_root = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops"
        if not os.path.exists(workspace_root) and "DATABRICKS_RUNTIME_VERSION" not in os.environ:
            cache_dir = "data"
        else:
            cache_dir = f"{workspace_root}/data"
            
        cache_path = os.path.join(cache_dir, ".last_processed_date.json")
        os.makedirs(cache_dir, exist_ok=True)
        
        cached_date = None
        last_processed_at_str = "Never"
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                    cached_date = cache_data.get("latest_date")
                    last_processed_at_str = cache_data.get("last_processed_at", "Never")
                logger.info(f"Cache loaded. Last processed date: {cached_date} at {last_processed_at_str}")
            except Exception as e:
                logger.warning(f"Could not read cache: {e}")
                
        # Determine if we have new data
        has_new_data = False
        if cached_date is None or latest_date_str > cached_date:
            has_new_data = True
            
        # Decision: run preprocessing or skip
        should_run = not check_freshness or has_new_data or force
        
        if should_run:
            logger.info("🚀 Running preprocessing...")
            
            # Call the preprocessing script with our paths
            preprocess(run_sorting=True, raw_path=raw_path, processed_path=os.path.dirname(output_folder) + "/")
            
            # Read the processed table to get the final row count
            df_processed = spark.read.parquet(output_folder)
            total_rows = df_processed.count()
            
            # Update cache file
            last_processed_at_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cache_update = {
                "latest_date": latest_date_str,
                "last_processed_at": last_processed_at_str
            }
            try:
                with open(cache_path, 'w') as f:
                    json.dump(cache_update, f, indent=2)
                logger.info(f"Cache updated with date: {latest_date_str}")
            except Exception as e:
                logger.warning(f"Could not write cache: {e}")
        else:
            logger.info("⏸️ Skipping preprocessing (no new data and force=False)")
            # Get row count from existing processed features
            try:
                df_processed = spark.read.parquet(output_folder)
                total_rows = df_processed.count()
            except Exception:
                total_rows = 0
                
        return {
            "success": True,
            "latest_date": latest_date_str,
            "total_rows": total_rows,
            "days_old": days_old,
            "has_new_data": has_new_data,
            "last_processed_at": last_processed_at_str
        }
        
    except Exception as e:
        logger.error(f"Error in databricks ingestion: {e}")
        import traceback
        return {
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}"
        }
