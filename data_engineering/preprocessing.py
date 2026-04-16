import os
import yaml
import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

def preprocess():
    raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    processed_path = "/Volumes/workspace/default/stock_data/processed/"
    
    spark = SparkSession.builder \
        .appName("StockPreprocessing") \
        .getOrCreate()

    # 1. Read raw Parquet files
    df = spark.read.parquet(raw_path)

    # 2. Clean data
    df = df.dropna().dropDuplicates()
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

    # Drop intermediate columns and rows with nulls (due to lag/lead)
    df = df.dropna()

    # 5. Normalization using StandardScaler
    feature_cols = ["open", "high", "low", "close", "volume", "return", "MA_5", "MA_10", "volatility"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
    df_vec = assembler.transform(df)

    scaler = StandardScaler(inputCol="features_vec", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(df_vec)
    df = scaler_model.transform(df_vec)

    # 6. Graph Dependency: Returns Correlation Matrix
    # Pivot to get returns for all tickers per date
    returns_df = df.groupBy("date").pivot("ticker").agg(F.first("return")).na.fill(0)
    ticker_cols = [c for c in returns_df.columns if c != 'date']
    
    # Assemble returns into a single vector column
    corr_assembler = VectorAssembler(inputCols=ticker_cols, outputCol="corr_features")
    corr_df = corr_assembler.transform(returns_df)
    
    # Pearson Correlation
    matrix = Correlation.corr(corr_df, "corr_features").head()[0]
    
    # Convert matrix to a DataFrame for saving (optional but good practice)
    # Here we just save the processed data and can export matrix as well
    
    # 7. Save outputs
    
    # Save processed data
    df.select("date", "ticker", "scaled_features", "target") \
      .write.mode("overwrite").parquet(processed_path + "stock_features.parquet")
    
    # Save correlation matrix
    matrix_array = matrix.toArray().tolist()
    corr_pdf = pd.DataFrame(matrix_array, columns=ticker_cols)
    corr_pdf.insert(0, "ticker", ticker_cols)
    corr_spark_df = spark.createDataFrame(corr_pdf)
    corr_spark_df.write.mode("overwrite").parquet(processed_path + "correlation_matrix.parquet")
    
if __name__ == "__main__":
    preprocess()
