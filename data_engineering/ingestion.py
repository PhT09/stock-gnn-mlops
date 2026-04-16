import yaml
import pandas as pd
from datetime import datetime, timedelta
import time
from pyspark.sql import SparkSession

from vnstock import Vnstock

def load_config():
    config_path = '/Workspace/Users/trannguyentoanphat1592005@gmail.com/config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ingest_data():
    config = load_config()
    tickers = config['tickers']
    years = config['params']['timeframe_years']
    
    spark = SparkSession.builder \
        .appName("StockIngestion") \
        .getOrCreate()
    
    raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    
    # Determine date range based on existing data
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Try to read existing data to get the last date
        existing_spark_df = spark.read.parquet(raw_path)
        existing_df = existing_spark_df.toPandas()
        
        if not existing_df.empty:
            # Get the maximum date from existing data
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            max_date = existing_df['date'].max()
            
            # Start from the day after the last date in existing data
            start_date = (max_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            print(f"Existing data found. Last date: {max_date.strftime('%Y-%m-%d')}")
            print(f"Fetching incremental data from {start_date} to {end_date}...")
            
            # Check if there's actually new data to fetch
            if start_date > end_date:
                print("Data is already up to date. No new data to fetch.")
                return
        else:
            # Empty dataframe - do full load
            start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
            print(f"Existing data is empty. Performing full load from {start_date} to {end_date}...")
            
    except Exception as e:
        # File doesn't exist - do full load
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        print(f"No existing data found ({str(e)}). Performing full load from {start_date} to {end_date}...")

    all_data = []

    print(f"Starting ingestion for {len(tickers)} tickers...")

    vn = Vnstock()
    for ticker in tickers:
        try:
            # vnstock historical data (Vnstock 3.x) using KBS source
            stock = vn.stock(symbol=ticker, source='KBS')
            df = stock.quote.history(start=start_date, end=end_date)
            
            if df is not None and not df.empty:
                # Standardize columns: date, ticker, open, high, low, close, volume
                # Vnstock 3.x returns time, open, high, low, close, volume
                df = df.rename(columns={'time': 'date'})
                df['ticker'] = ticker
                df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
                all_data.append(df)
                print(f"Successfully fetched {ticker}: {len(df)} rows")
            else:
                print(f"No data for {ticker}")
            
            # Rate limit handling: Wait 3 seconds between requests (20 requests/min limit)
            time.sleep(3)
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            # If hit limit, wait longer
            if "Rate limit" in str(e) or "limit" in str(e).lower():
                print("Rate limit potentially hit. Sleeping for 60 seconds...")
                time.sleep(60)


    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        # Ensure date is datetime type and compatible with Spark (microseconds)
        new_df['date'] = pd.to_datetime(new_df['date']).dt.floor('us')
        
        print(f"Fetched {len(new_df)} new rows. Merging with existing data...")

        # Try to read existing data again for merging
        try:
            existing_spark_df = spark.read.parquet(raw_path)
            existing_df = existing_spark_df.toPandas()
            print(f"Found existing data with {len(existing_df)} rows")
            
            # Combine new data with existing data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates based on date and ticker, keeping the latest entry
            combined_df = combined_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
            combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
            
            print(f"After merge and deduplication: {len(combined_df)} rows")
            final_df = combined_df
        except Exception as e:
            # File doesn't exist yet or error reading - use new data only
            print(f"No existing data found during merge ({str(e)}). Using new data only.")
            final_df = new_df
        
        # Write merged data back to Volume
        final_spark_df = spark.createDataFrame(final_df)
        final_spark_df.write.mode("overwrite").parquet(raw_path)
                
        print(f"Data saved successfully to Volume: {raw_path}")
        print(f"Total rows in dataset: {len(final_df)}")
    else:
        print("No new data fetched.")

if __name__ == "__main__":
    ingest_data()
