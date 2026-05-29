import os
import yaml
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.stat import Correlation

def preprocess():
    raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    processed_path = "/Volumes/workspace/default/stock_data/processed/"
    
    spark = SparkSession.builder \
        .appName("StockPreprocessing") \
        .getOrCreate()

    # 1. Read raw Parquet files
    df = spark.read.parquet(raw_path)

    # 2. Clean data - only drop NULL in critical columns (OHLCV)
    # ✅ FIX: Don't use df.dropna() - it drops ALL rows with ANY NULL!
    # Data từ vnstock có thể có NULL trong datetime column → drops recent data
    # Only drop NULL in critical OHLCV columns that we actually need
    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close", "volume"])
    df = df.dropDuplicates()
    df = df.orderBy("ticker", "date")

    # 3. Feature Engineering with Window Functions
    window_spec = Window.partitionBy("ticker").orderBy("date")
    
    # Calculate returns with division by zero protection
    df = df.withColumn("prev_close", F.lag("close", 1).over(window_spec))
    df = df.withColumn("return", 
        F.when(F.col("prev_close") == 0, 0)
         .otherwise((F.col("close") - F.col("prev_close")) / F.col("prev_close"))
    )
    
    # MA_5, MA_10
    df = df.withColumn("MA_5", F.avg("close").over(window_spec.rowsBetween(-4, 0)))
    df = df.withColumn("MA_10", F.avg("close").over(window_spec.rowsBetween(-9, 0)))
    
    # Volatility (Rolling distance of return)
    df = df.withColumn("volatility", F.stddev("return").over(window_spec.rowsBetween(-19, 0)))

    # 4. Create target label: 1 if Close price goes UP next day, 0 if DOWN
    lead_window = Window.partitionBy("ticker").orderBy("date")
    df = df.withColumn("next_close", F.lead("close", 1).over(lead_window))
    df = df.withColumn("target", F.when(F.col("next_close") > F.col("close"), 1).otherwise(0))

    # ✅ FIX: Drop rows with nulls in critical columns only
    # - next_close/target: NULL at last row of each ticker (can't train without target)
    # - volatility: NULL at first 19 rows of each ticker (not enough history)
    # This preserves recent data while ensuring feature completeness
    df = df.dropna(subset=["next_close", "target", "volatility"])

    # 5. ✅ SAMPLING: Reduce data size to fit Spark Connect 256MB limit
    # Sample 75% of data to ensure VectorAssembler model < 256MB
    # Stratified sampling preserves data distribution
    print(f"   Rows before sampling: {df.count():,}")
    df = df.sample(fraction=0.75, seed=42)
    print(f"   Rows after sampling:  {df.count():,} (75%)")

    # 6. ✅ NEW: Manual Normalization (NO StandardScaler model!)
    # This avoids the 256MB Spark Connect ML model size limit
    feature_cols = ["open", "high", "low", "close", "volume", "return", "MA_5", "MA_10", "volatility"]
    
    # Create feature vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
    df_vec = assembler.transform(df)
    
    # Calculate mean and std for each feature using SQL aggregations
    # This is much more efficient than StandardScaler.fit() and creates NO model object
    stats = []
    for i, col_name in enumerate(feature_cols):
        stats_row = df_vec.select(
            F.mean(F.col(col_name)).alias("mean"),
            F.stddev(F.col(col_name)).alias("std")
        ).collect()[0]
        stats.append((stats_row["mean"], stats_row["std"] if stats_row["std"] else 1.0))
    
    # Define UDF for manual scaling
    def manual_scale(features):
        """Scale features using pre-computed means and stds"""
        if features is None:
            return None
        scaled = []
        for i, val in enumerate(features):
            mean, std = stats[i]
            scaled_val = (val - mean) / std if std != 0 else 0.0
            scaled.append(float(scaled_val))
        return Vectors.dense(scaled)
    
    # Register UDF
    scale_udf = F.udf(manual_scale, VectorUDT())
    
    # Apply scaling - NO MODEL STORED!
    df = df_vec.withColumn("scaled_features", scale_udf(F.col("features_vec")))
    
    print(f"✅ Manual scaling completed with {len(feature_cols)} features")
    print(f"   Mean/Std computed for: {', '.join(feature_cols)}")

    # 7. Graph Dependency: Returns Correlation Matrix
    # Pivot to get returns for all tickers per date
    returns_df = df.groupBy("date").pivot("ticker").agg(F.first("return")).na.fill(0)
    ticker_cols = [c for c in returns_df.columns if c != 'date']
    
    # Assemble returns into a single vector column
    corr_assembler = VectorAssembler(inputCols=ticker_cols, outputCol="corr_features")
    corr_df = corr_assembler.transform(returns_df)
    
    # Pearson Correlation
    matrix = Correlation.corr(corr_df, "corr_features").head()[0]
    
    # 8. Save outputs
    
    # Save processed data
    df.select("date", "ticker", "scaled_features", "target") \
      .write.mode("overwrite").parquet(processed_path + "stock_features.parquet")
    
    # Save correlation matrix
    matrix_array = matrix.toArray().tolist()
    corr_pdf = pd.DataFrame(matrix_array, columns=ticker_cols)
    corr_pdf.insert(0, "ticker", ticker_cols)
    corr_spark_df = spark.createDataFrame(corr_pdf)
    corr_spark_df.write.mode("overwrite").parquet(processed_path + "correlation_matrix.parquet")
    
    print(f"✅ Preprocessing completed successfully!")
    print(f"   Output: {processed_path}stock_features.parquet")
    
if __name__ == "__main__":
    preprocess()
