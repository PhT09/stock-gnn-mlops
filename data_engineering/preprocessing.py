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
    """Engineer features exactly matching the user's reference code
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: DataFrame with 16 engineered features
    """
    logger.info("[FEATURE ENGINEERING]")
    
    window_spec = Window.partitionBy("ticker").orderBy("date")
    
    # Helper lag columns for computations
    df = df.withColumn("prev_close_1", F.lag("close", 1).over(window_spec))
    df = df.withColumn("prev_close_3", F.lag("close", 3).over(window_spec))
    df = df.withColumn("prev_close_5", F.lag("close", 5).over(window_spec))
    df = df.withColumn("prev_close_10", F.lag("close", 10).over(window_spec))
    df = df.withColumn("prev_volume_1", F.lag("volume", 1).over(window_spec))
    
    # 1. Return Features
    df = df.withColumn("return_1d", F.when(df.prev_close_1 == 0, 0.0).otherwise((df.close - df.prev_close_1) / df.prev_close_1))
    df = df.withColumn("return_3d", F.when(df.prev_close_3 == 0, 0.0).otherwise((df.close - df.prev_close_3) / df.prev_close_3))
    df = df.withColumn("return_5d", F.when(df.prev_close_5 == 0, 0.0).otherwise((df.close - df.prev_close_5) / df.prev_close_5))
    df = df.withColumn("return_10d", F.when(df.prev_close_10 == 0, 0.0).otherwise((df.close - df.prev_close_10) / df.prev_close_10))
    
    # 2. Moving Average Features
    df = df.withColumn("ma5", F.avg("close").over(window_spec.rowsBetween(-4, 0)))
    df = df.withColumn("ma10", F.avg("close").over(window_spec.rowsBetween(-9, 0)))
    
    df = df.withColumn("price_vs_ma5", F.when(df.ma5 == 0, 1.0).otherwise(df.close / df.ma5))
    df = df.withColumn("price_vs_ma10", F.when(df.ma10 == 0, 1.0).otherwise(df.close / df.ma10))
    df = df.withColumn("ma5_vs_ma10", F.when(df.ma10 == 0, 1.0).otherwise(df.ma5 / df.ma10))
    
    # 3. Volume Features
    df = df.withColumn("avg_vol20", F.avg("volume").over(window_spec.rowsBetween(-19, 0)))
    df = df.withColumn("volume_ratio", F.when(df.avg_vol20 == 0, 1.0).otherwise(df.volume / df.avg_vol20))
    df = df.withColumn("volume_change", F.when(df.prev_volume_1 == 0, 0.0).otherwise((df.volume - df.prev_volume_1) / df.prev_volume_1))
    
    # 4. Volatility Features
    df = df.withColumn("volatility_5", F.stddev("return_1d").over(window_spec.rowsBetween(-4, 0)))
    df = df.withColumn("volatility_10", F.stddev("return_1d").over(window_spec.rowsBetween(-9, 0)))
    
    # 5. Intraday Features
    df = df.withColumn("oc_return", F.when(df.open == 0, 0.0).otherwise(df.close / df.open - 1.0))
    df = df.withColumn("hl_range", F.when(df.close == 0, 0.0).otherwise((df.high - df.low) / df.close))
    df = df.withColumn("close_position", 
                       F.when((df.high - df.low) == 0, 0.5)
                        .otherwise((df.close - df.low) / (df.high - df.low)))
    
    # 6. Lag Features
    df = df.withColumn("return_lag1", F.lag("return_1d", 1).over(window_spec))
    df = df.withColumn("return_lag2", F.lag("return_1d", 2).over(window_spec))
    df = df.withColumn("return_lag3", F.lag("return_1d", 3).over(window_spec))
    
    # Drop temporary helper columns
    temp_cols = ["prev_close_1", "prev_close_3", "prev_close_5", "prev_close_10", "prev_volume_1", "ma5", "ma10", "avg_vol20"]
    df = df.drop(*temp_cols)
    
    logger.info("  16 features engineered successfully")
    return df

def create_target_labels(df):
    """Create target labels for prediction (1=price up, 0=price down, null=latest row)
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: DataFrame with target labels
        int: Row count
    """
    logger.info("[TARGET LABEL CREATION]")
    
    logger.info("Creating target labels (1=price up, 0=price down, null=latest)...")
    lag_window = Window.partitionBy("ticker").orderBy("date")
    df = df.withColumn("next_close", F.lead("close", 1).over(lag_window))
    
    # Target is null if next_close is null (latest day's data), otherwise 1 or 0
    df = df.withColumn("target", 
        F.when(F.col("next_close").isNull(), F.lit(None).cast("integer"))
         .when(F.col("next_close") > F.col("close"), 1)
         .otherwise(0))
         
    df = df.drop("next_close")
    row_count = df.count()
    logger.info(f"  Target labels created. Total rows: {row_count:,}")
    
    # Check target distribution (ignoring nulls)
    logger.info("Calculating target label distribution (excluding latest day)...")
    target_dist = df.dropna(subset=["target"]).groupBy("target").count().collect()
    total_labeled = sum(row['count'] for row in target_dist)
    logger.info("Target label distribution:")
    for row in target_dist:
        label = "UP" if row['target'] == 1 else "DOWN"
        count = row['count']
        pct = count / total_labeled * 100 if total_labeled > 0 else 0
        logger.info(f"  - {label} (target={row['target']}): {count:,} ({pct:.2f}%)")
    
    return df, row_count

def normalize_features(df, feature_cols):
    """Normalize features using Z-score normalization
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names to normalize
        
    Returns:
        DataFrame: DataFrame with normalized features and scaled_features VectorUDT column
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
    
    # Assemble scaled features into a single VectorUDT column for training compatibility
    logger.info("Assembling scaled features into VectorUDT...")
    assembler = VectorAssembler(inputCols=scaled_feature_cols, outputCol="scaled_features")
    df = assembler.transform(df)
    logger.info("  ✓ Features assembled into scaled_features column")
    
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
    columns_to_save = ["date", "ticker", "scaled_features", "target"]
    
    # Add original feature columns
    columns_to_save.extend(feature_cols)
    
    # Add scaled feature columns
    scaled_cols = [f"{col}_scaled" for col in feature_cols]
    columns_to_save.extend(scaled_cols)
    
    # Add close column for visualization/reporting if available
    if "close" in df.columns:
        columns_to_save.append("close")
    
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

def sort_price(spark=None, raw_path=None, output_dir=None):
    logger.info("[SORTING STOCKS BY PRICE]")
    
    # Initialize Spark if not provided
    if spark is None:
        spark = initialize_spark("SortStocksByPrice")
    
    # Set default paths
    if raw_path is None:
        raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    if output_dir is None:
        output_dir = "/Volumes/workspace/default/stock_data/processed/"
    
    # Get current date and session
    today = datetime.now().date().strftime('%Y-%m-%d')
    current_session = get_current_session()
    logger.info(f"Current date: {today}")
    logger.info(f"Current session: {current_session}")
    
    # Read raw data
    df_raw = spark.read.parquet(raw_path)
    
    # Filter by today's date and session
    session_pattern = f"{today}{current_session}"
    df_today = df_raw.filter(F.col("date") == session_pattern)
    
    # Sort by close price descending
    df_sorted = df_today.select("ticker", "close", "date").orderBy(F.desc("close"))
    
    # Collect results
    sorted_list = df_sorted.collect()
    
    logger.info(f"Found {len(sorted_list)} stocks for session {current_session}")
    
    # Save to file
    output_file = output_dir + "stock_desc_price.csv"
    logger.info(f"Saving sorted stocks to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ticker, close\n")
        for row in sorted_list:
            f.write(f"{row['ticker']}, {row['close']:.2f}\n")
    
    logger.info(f"  Saved {len(sorted_list)} stocks to {output_file}")
    
    return sorted_list

def sort_volume(spark=None, raw_path=None, output_dir=None):
    logger.info("[SORTING STOCKS BY VOLUME]")
    
    # Initialize Spark if not provided
    if spark is None:
        spark = initialize_spark("SortStocksByVolume")
    
    # Set default paths
    if raw_path is None:
        raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    if output_dir is None:
        output_dir = "/Volumes/workspace/default/stock_data/processed/"
    
    # Get current date and session
    today = datetime.now().date().strftime('%Y-%m-%d')
    current_session = get_current_session()
    logger.info(f"Current date: {today}")
    logger.info(f"Current session: {current_session}")
    
    # Read raw data
    df_raw = spark.read.parquet(raw_path)
    
    # Filter by today's date and session
    session_pattern = f"{today}{current_session}"
    df_today = df_raw.filter(F.col("date") == session_pattern)
    
    # Sort by volume descending
    df_sorted = df_today.select("ticker", "volume", "date").orderBy(F.desc("volume"))
    
    # Collect results
    sorted_list = df_sorted.collect()
    
    logger.info(f"Found {len(sorted_list)} stocks for session {current_session}")
    
    # Save to file
    output_file = output_dir + "stock_desc_volume.csv"
    logger.info(f"Saving sorted stocks to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ticker, volume\n")
        for row in sorted_list:
            f.write(f"{row['ticker']}, {row['volume']:,}\n")
    
    logger.info(f"  Saved {len(sorted_list)} stocks to {output_file}")
    
    return sorted_list

# MAIN PREPROCESSING FUNCTION

def preprocess(run_sorting=True, raw_path=None, processed_path=None):
    """Main preprocessing function
    
    Args:
        run_sorting: Whether to run sorting functions after preprocessing (default: True)
        raw_path: Custom raw data path (default: volume path)
        processed_path: Custom processed data path (default: volume path)
    """
    logger.info("STOCK DATA PREPROCESSING PIPELINE")
    
    # Paths
    if raw_path is None:
        raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    if processed_path is None:
        processed_path = "/Volumes/workspace/default/stock_data/processed/"
    
    # Initialize Spark
    spark = initialize_spark("StockPreprocessing")
    
    # Step 1: Read raw data
    df, initial_count, ticker_count = read_raw_data(spark, raw_path)
    
    # Step 2: Clean data
    df = clean_data(df, initial_count)
    
    # Step 3: Engineer features
    df = engineer_features(df)
    
    # Step 4: Create target labels
    df, after_windowing = create_target_labels(df)
    
    # Step 5: Normalize features
    feature_cols = [
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "price_vs_ma5",
        "price_vs_ma10",
        "ma5_vs_ma10",
        "volume_ratio",
        "volume_change",
        "volatility_5",
        "volatility_10",
        "oc_return",
        "hl_range",
        "close_position",
        "return_lag1",
        "return_lag2",
        "return_lag3"
    ]
    
    # Drop rows with nulls in features before normalization
    df = df.dropna(subset=feature_cols)
    
    df = normalize_features(df, feature_cols)
    
    # Step 6: Save processed data
    final_count = save_processed_data(df, processed_path, feature_cols)
    
    # Step 7: Sort by price and volume (if requested)
    if run_sorting:
        logger.info("\n[RUNNING SORTING OPERATIONS]")
        sort_price(spark, raw_path=raw_path, output_dir=processed_path)
        sort_volume(spark, raw_path=raw_path, output_dir=processed_path)
    
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
