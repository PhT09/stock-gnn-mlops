import os
import yaml
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# UTILITY FUNCTIONS

def get_current_session():
    """Determine current trading session based on Vietnam time (UTC+7)
    Returns:
        - '(1)' for morning session (9:00 - 11:30)
        - '(2)' for afternoon session (13:30 - 15:00)
        - '(2)' as default for outside trading hours
    """
    vietnam_tz = timezone(timedelta(hours=7))
    now = datetime.now(tz=vietnam_tz)
    current_time = now.time()
    
    morning_start = datetime.strptime("09:00", "%H:%M").time()
    afternoon_start = datetime.strptime("13:30", "%H:%M").time()
    
    if morning_start <= current_time < afternoon_start:
        return '(1)'
    else:
        return '(2)'

def initialize_spark(app_name="StockPreprocessing"):
    """Initialize Spark session"""
    return SparkSession.builder.appName(app_name).getOrCreate()

# DATA LOADING FUNCTIONS

def read_raw_data(spark, raw_path):
    """Read raw data from parquet file
    
    Args:
        spark: SparkSession
        raw_path: Path to raw data parquet file
        
    Returns:
        DataFrame: Raw data
        int: Initial row count
        int: Unique ticker count
    """
    logger.info("[READING RAW DATA]")
    logger.info(f"Reading from: {raw_path}")
    
    df = spark.read.parquet(raw_path)
    initial_count = df.count()
    ticker_count = df.select("ticker").distinct().count()
    
    logger.info("Raw data loaded successfully")
    logger.info(f"  - Total rows: {initial_count:,}")
    logger.info(f"  - Unique tickers: {ticker_count}")
    logger.info(f"  - Columns: {len(df.columns)}")
    
    return df, initial_count, ticker_count

# DATA CLEANING FUNCTIONS

def format_date_column(df):
    """Format date column to keep session markers but remove time
    
    Transforms:
    - '2026-06-01(2)' -> '2026-06-01(2)' (keep as is)
    - '2026-06-01 07:00:00' -> '2026-06-01' (remove time, no session marker)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: DataFrame with formatted date column
    """
    logger.info("Formatting date column (keep session markers, remove time)...")
    
    # Extract session marker if exists: (1) or (2)
    df = df.withColumn("session_marker",
        F.when(F.col("date").rlike(r"\(1\)"), "(1)")
         .when(F.col("date").rlike(r"\(2\)"), "(2)")
         .otherwise(""))
    
    # Extract date portion only (YYYY-MM-DD)
    # Remove anything after the date: space, time, or session marker
    df = df.withColumn("date_only", 
        F.regexp_extract(F.col("date"), r"(\d{4}-\d{2}-\d{2})", 1))
    
    # Combine date + session marker
    df = df.withColumn("date",
        F.concat(F.col("date_only"), F.col("session_marker")))
    
    # Drop temporary columns
    df = df.drop("session_marker", "date_only")
    
    # Sample output for verification
    sample_dates = df.select("date").distinct().limit(5).collect()
    logger.info("  Sample formatted dates:")
    for row in sample_dates:
        logger.info(f"    - {row['date']}")
    
    return df

def clean_data(df, initial_count):
    """Clean data by removing nulls and duplicates
    
    Args:
        df: Input DataFrame
        initial_count: Initial row count for logging
        
    Returns:
        DataFrame: Cleaned data
    """
    logger.info("[DATA CLEANING]")
    
    # Format date column first
    df = format_date_column(df)
    
    # Remove null values
    logger.info("Removing null values...")
    df = df.dropna()
    after_dropna = df.count()
    logger.info(f"  - Rows after dropna: {after_dropna:,} (removed {initial_count - after_dropna:,})")
    
    # Remove duplicates
    logger.info("Removing duplicates...")
    df = df.dropDuplicates()
    after_dedup = df.count()
    logger.info(f"  - Rows after dedup: {after_dedup:,} (removed {after_dropna - after_dedup:,})")
    
    # Sort by ticker and date
    logger.info("Sorting by ticker and date...")
    df = df.orderBy("ticker", "date")
    logger.info("Data cleaning completed")
    
    return df

# FEATURE ENGINEERING FUNCTIONS

def engineer_features(df):
    """Engineer features: returns, moving averages, volatility
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: DataFrame with engineered features
    """
    logger.info("[FEATURE ENGINEERING]")
    
    window_spec = Window.partitionBy("ticker").orderBy("date")
    
    # Calculate returns
    logger.info("Calculating returns...")
    df = df.withColumn("prev_close", F.lag("close", 1).over(window_spec))
    df = df.withColumn("return", 
        F.when(F.col("prev_close") == 0, 0)
         .otherwise((F.col("close") - F.col("prev_close")) / F.col("prev_close"))
    )
    logger.info("  Returns calculated")
    
    # Moving averages
    logger.info("Calculating moving averages (MA_5, MA_10)...")
    df = df.withColumn("MA_5", F.avg("close").over(window_spec.rowsBetween(-4, 0)))
    df = df.withColumn("MA_10", F.avg("close").over(window_spec.rowsBetween(-9, 0)))
    logger.info("  Moving averages calculated")
    
    # Volatility
    logger.info("Calculating volatility (20-day rolling std of returns)...")
    df = df.withColumn("volatility", F.stddev("return").over(window_spec.rowsBetween(-19, 0)))
    logger.info(" Volatility calculated")
    
    return df

def create_target_labels(df):
    """Create target labels for prediction
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: DataFrame with target labels
        int: Row count after windowing operations
    """
    logger.info("[TARGET LABEL CREATION]")
    
    logger.info("Creating target labels (1=price up, 0=price down)...")
    lag_window = Window.partitionBy("ticker").orderBy("date")
    df = df.withColumn("prev_close", F.lag("close", 1).over(lag_window))
    df = df.withColumn("target", 
        F.when(F.col("close") > F.col("prev_close"), 1)
         .otherwise(0))
    logger.info("  Target labels created")
    
    # Drop rows with nulls (due to lag/lead)
    logger.info("Removing first day of each ticker (no previous price)...")
    before_dropna = df.count()
    df = df.filter(F.col("prev_close").isNotNull())
    df = df.drop("prev_close")
    after_windowing = df.count()
    logger.info(f"  Rows after filtering: {after_windowing:,} (removed {before_dropna - after_windowing:,})")
    
    # Check target distribution
    logger.info("Calculating target label distribution...")
    target_dist = df.groupBy("target").count().collect()
    logger.info("Target label distribution:")
    for row in target_dist:
        label = "UP" if row['target'] == 1 else "DOWN"
        count = row['count']
        pct = count / after_windowing * 100
        logger.info(f"  - {label} (target={row['target']}): {count:,} ({pct:.2f}%)")
    
    return df, after_windowing

def normalize_features(df, feature_cols):
    """Normalize features using Z-score normalization
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names to normalize
        
    Returns:
        DataFrame: DataFrame with normalized features
    """
    logger.info("[FEATURE NORMALIZATION]")
    
    logger.info(f"Features to normalize: {', '.join(feature_cols)}")
    logger.info("Using SQL-based Z-score normalization (mean=0, std=1)...")
    
    # Calculate mean and stddev for each feature column globally
    logger.info("Computing statistics for normalization...")
    stats = {}
    for col_name in feature_cols:
        stats[col_name] = df.select(
            F.mean(col_name).alias('mean'),
            F.stddev(col_name).alias('std')
        ).collect()[0]
        logger.info(f"  - {col_name}: mean={stats[col_name]['mean']:.4f}, std={stats[col_name]['std']:.4f}")
    
    # Apply normalization: (x - mean) / std
    logger.info("Applying normalization...")
    for col_name in feature_cols:
        mean_val = stats[col_name]['mean']
        std_val = stats[col_name]['std']
        
        # Avoid division by zero
        if std_val > 0:
            df = df.withColumn(
                f"{col_name}_scaled",
                (F.col(col_name) - mean_val) / std_val
            )
        else:
            df = df.withColumn(f"{col_name}_scaled", F.lit(0.0))
    
    scaled_feature_cols = [f"{col}_scaled" for col in feature_cols]
    logger.info(f"  ✓ {len(scaled_feature_cols)} features normalized")
    
    return df

# DATA SAVING FUNCTIONS

def save_processed_data(df, processed_path, feature_cols):
    """Save processed data to parquet file
    
    Args:
        df: DataFrame to save
        processed_path: Base path for processed data
        feature_cols: List of original feature column names
        
    Returns:
        int: Final row count
    """
    logger.info("[SAVING PROCESSED DATA]")
    
    # Build list of columns to save
    columns_to_save = ["date", "ticker"]
    
    # Add original feature columns
    columns_to_save.extend(feature_cols)
    
    # Add scaled feature columns
    scaled_cols = [f"{col}_scaled" for col in feature_cols]
    columns_to_save.extend(scaled_cols)
    
    # Add target column
    columns_to_save.append("target")
    
    logger.info(f"Columns to save: {', '.join(columns_to_save)}")
    
    # Save processed features
    feature_output_path = processed_path + "stock_features.parquet"
    logger.info(f"Saving processed features to: {feature_output_path}")
    df.select(*columns_to_save) \
      .write.mode("overwrite").parquet(feature_output_path)
    final_count = df.count()
    logger.info(f"  Features saved: {final_count:,} rows")
    
    return final_count

# ============================================================================
# SORTING AND EXPORT FUNCTIONS
# ============================================================================

def export_latest_price_volume(spark=None, raw_path=None, output_dir=None):
    logger.info("[EXPORTING LATEST PRICE & VOLUME PER TICKER]")
    
    # Initialize Spark if not provided
    if spark is None:
        spark = initialize_spark("ExportLatestPriceVolume")
    
    # Set default paths
    if raw_path is None:
        raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    if output_dir is None:
        output_dir = "/Volumes/workspace/default/stock_data/processed/"
    
    # Read raw data
    df_raw = spark.read.parquet(raw_path)
    
    # Find latest session per ticker
    window_spec = Window.partitionBy("ticker").orderBy(F.col("date").desc())
    df_latest = df_raw.withColumn("rn", F.row_number().over(window_spec)) \
                     .filter(F.col("rn") == 1) \
                     .select("ticker", "close", "volume", "date")
    
    # Save to file
    output_file = output_dir + "ticker_price_volume.csv"
    logger.info(f"Saving latest price & volume to: {output_file}")
    
    # Convert to Pandas for CSV export
    df_pd = df_latest.toPandas()
    df_pd.to_csv(output_file, index=False, columns=["ticker", "close", "volume"])
    
    logger.info(f"  Saved {len(df_pd)} tickers to {output_file}")
    
    return df_latest

# MAIN PREPROCESSING FUNCTION

def preprocess(run_sorting=True):
    """Main preprocessing function
    
    Args:
        run_sorting: Whether to run sorting functions after preprocessing (default: True)
    """
    logger.info("STOCK DATA PREPROCESSING PIPELINE")
    
    # Paths
    raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    processed_path = "/Volumes/workspace/default/stock_data/processed/"
    
    # Initialize Spark
    spark = initialize_spark("StockPreprocessing")
    
    # Step 1: Read raw data
    df, initial_count, ticker_count = read_raw_data(spark, raw_path)
    
    # Step 2: Clean data
    df = clean_data(df, initial_count)
    
    # Step 3: Engineer features
    # df = engineer_features(df)
    
    # Step 4: Create target labels
    df, after_windowing = create_target_labels(df)
    
    # Step 5: Normalize features
    feature_cols = ["open", "high", "low", "close", "volume"]
    df = normalize_features(df, feature_cols)
    
    # Step 6: Save processed data
    final_count = save_processed_data(df, processed_path, feature_cols)
    
    # Step 7: Sort by price and volume (if requested)
    if run_sorting:
        export_latest_price_volume(spark)
    
    # Summary
    
    logger.info("PREPROCESSING SUMMARY")
    logger.info(f"Initial rows:      {initial_count:,}")
    logger.info(f"Final rows:        {final_count:,}")
    logger.info(f"Rows removed:      {initial_count - final_count:,}")
    logger.info(f"Unique tickers:    {ticker_count}")
    logger.info(f"Features per row:  {len(feature_cols)}")
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    preprocess()
