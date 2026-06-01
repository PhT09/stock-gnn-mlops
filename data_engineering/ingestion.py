import os
import yaml
import pandas as pd
from datetime import datetime, timedelta, time as dt_time, timezone
import time
from pyspark.sql import SparkSession
from dotenv import load_dotenv

from vnstock.api.quote import Quote

def load_config():
    """Load configuration from YAML file"""
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_api_key():
    """Load VNSTOCK API key from .env file"""
    env_path = '.env'
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        api_key = os.getenv('VNSTOCK_API')
        
        if api_key:
            print(f"VNSTOCK API key loaded from .env file")
            # Set as environment variable for vnstock to use
            os.environ['VNSTOCK_API_KEY'] = api_key
            return api_key
        else:
            print("VNSTOCK_API not found in .env file. Using guest mode.")
            return None
    else:
        print(f".env file not found at {env_path}. Using guest mode.")
        return None

def get_trading_session():
    """Determine current trading session based on Vietnam time (UTC+7)
    Returns:
        - '(1)' for morning session (9:00 - 11:30)
        - '(2)' for afternoon session (13:30 - 15:00)
        - '(2)' as default for outside trading hours
    """
    # Vietnam timezone is UTC+7
    vietnam_tz = timezone(timedelta(hours=7))
    now = datetime.now(tz=vietnam_tz)
    current_time = now.time()
    
    # Morning session: 9:00 - 11:30
    morning_start = datetime.strptime("09:00", "%H:%M").time()
    morning_end = datetime.strptime("11:30", "%H:%M").time()
    
    # Afternoon session: 13:30 - 15:00
    afternoon_start = datetime.strptime("13:30", "%H:%M").time()
    afternoon_end = datetime.strptime("15:00", "%H:%M").time()
    
    if morning_start <= current_time < afternoon_start:
        return '(1)', current_time, morning_start, afternoon_start
    elif afternoon_start <= current_time:
        return '(2)', current_time, afternoon_start
    else:
        # Default to afternoon session if outside trading hours
        return '(2)', current_time, morning_start, morning_end, afternoon_start, afternoon_end

def check_existing_data(spark, raw_path):
    """Check if data file exists and return last date if available
    Returns:
        tuple: (file_exists: bool, last_date: datetime.date or None, existing_df: pd.DataFrame or None)
    """
    try:
        existing_spark_df = spark.read.parquet(raw_path)
        existing_df = existing_spark_df.toPandas()
        
        if not existing_df.empty:
            # Parse date column (remove session markers if present)
            existing_df['date_clean'] = existing_df['date'].astype(str).str.replace(r'\(\d+\)', '', regex=True)
            existing_df['date_parsed'] = pd.to_datetime(existing_df['date_clean'])
            max_date = existing_df['date_parsed'].max().date()
            
            print(f"Existing data found with {len(existing_df)} rows. Last date: {max_date}")
            return True, max_date, existing_df
        else:
            print("Existing data file found but empty.")
            return True, None, existing_df
            
    except Exception as e:
        print(f"No existing data found ({str(e)}). Will create new file.")
        return False, None, None

def determine_fetch_strategy(file_exists, last_date, years):
    """Determine what data to fetch based on existing data
    Returns:
        tuple: (fetch_history: bool, fetch_intraday: bool, history_start: str, history_end: str)
    """
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    if not file_exists or last_date is None:
        # File doesn't exist or empty - fetch 5 years history + today's intraday
        history_start = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        print(f"Strategy: Full load - history from {history_start} to {yesterday_str} + today's intraday")
        return True, True, history_start, yesterday_str
    
    if last_date == yesterday:
        # Last date is yesterday - only fetch today's intraday
        print(f"Strategy: Data up to date through yesterday. Fetching only today's intraday data")
        return False, True, None, None
    
    elif last_date < yesterday:
        # Gap detected - fetch history from last date to yesterday + today's intraday
        history_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Strategy: Gap detected. Fetching history from {history_start} to {yesterday_str} + today's intraday")
        return True, True, history_start, yesterday_str
    
    else:
        # Last date is today or future - only fetch today's intraday to update
        print(f"Strategy: Data includes today. Fetching today's intraday to update")
        return False, True, None, None

def fetch_historical_data(ticker, start_date, end_date):
    """Fetch historical data for a ticker using Quote.history()
    Returns:
        pd.DataFrame or None
    """
    try:
        # Use new API: Quote(symbol, source)
        quote = Quote(symbol=ticker, source='KBS')
        df_history = quote.history(start=start_date, end=end_date)
        
        if df_history is not None and not df_history.empty:
            # Standardize columns
            df_history = df_history.rename(columns={'time': 'date'})
            df_history['ticker'] = ticker
            df_history = df_history[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
            # Ensure date column is string (no session marker for historical data)
            df_history['date'] = df_history['date'].astype(str)
            print(f"  {ticker} - History: {len(df_history)} rows from {start_date} to {end_date}")
            return df_history
        else:
            print(f"  {ticker} - No historical data")
            return None
            
    except (Exception, SystemExit) as e:
        print(f"  {ticker} - Error fetching history: {str(e)}")
        # If hit limit, wait longer
        if "rate limit" in str(e).lower() or "systemexit" in str(type(e).__name__).lower():
            print("Rate limit hit. Sleeping for 60 seconds before continuing...")
            time.sleep(60)
        return None

def fetch_intraday_data(ticker, session_marker):
    """Fetch today's intraday data for a ticker using Quote.intraday()
    Filters data by trading session and calculates OHLCV for that session
    
    Args:
        ticker: Stock ticker symbol
        session_marker: '(1)' for morning or '(2)' for afternoon
    
    Returns:
        pd.DataFrame or None
    """
    try:
        # Use new API: Quote(symbol, source)
        quote = Quote(symbol=ticker, source='KBS')
        df_intraday = quote.intraday(page_size=10000)
        
        if df_intraday is not None and not df_intraday.empty:
            # Standardize columns
            df_intraday = df_intraday.rename(columns={'time': 'date'})
            
            # Parse time from date column to filter by session
            df_intraday['datetime'] = pd.to_datetime(df_intraday['date'])
            df_intraday['time'] = df_intraday['datetime'].dt.time
            
            # Define session time ranges
            if session_marker == '(1)':
                # Morning session: 9:00 - 11:30
                session_start = dt_time(9, 0)
                session_end = dt_time(11, 30)
            else:
                # Afternoon session: 13:30 - 15:00
                session_start = dt_time(13, 30)
                session_end = dt_time(15, 0)
            
            # Filter data by session
            df_session = df_intraday[
                (df_intraday['time'] >= session_start) & 
                (df_intraday['time'] <= session_end)
            ].copy()
            
            if df_session.empty:
                print(f"  {ticker} - No intraday data for session {session_marker}")
                return None
            
            # Calculate OHLCV for the session
            # Get the date (without time) for the session
            session_date = df_session['datetime'].dt.date.iloc[0]
            
            # Calculate session OHLCV
            ohlcv_data = {
                'date': f"{session_date}{session_marker}",
                'ticker': ticker,
                'open': df_session['price'].iloc[0],  # First price in session
                'high': df_session['price'].max(),     # Highest price in session
                'low': df_session['price'].min(),       # Lowest price in session
                'close': df_session['price'].iloc[-1], # Last price in session
                'volume': df_session['volume'].sum()   # Total volume in session
            }
            
            # Create single-row DataFrame for this session
            result_df = pd.DataFrame([ohlcv_data])
            
            print(f"  {ticker} - Intraday: 1 row for session {session_marker} (aggregated from {len(df_session)} ticks)")
            return result_df
        else:
            print(f"  {ticker} - No intraday data")
            return None
            
    except (Exception, SystemExit) as e:
        print(f"  {ticker} - Error fetching intraday: {str(e)}")
        # If hit limit, wait longer
        if "rate limit" in str(e).lower() or "systemexit" in str(type(e).__name__).lower():
            print("Rate limit hit. Sleeping for 60 seconds before continuing...")
            time.sleep(60)
        return None

def merge_and_save_data(spark, raw_path, new_df, existing_df):
    """Merge new data with existing data and save to parquet
    """
    if existing_df is not None and not existing_df.empty:
        print(f"Merging {len(new_df)} new rows with {len(existing_df)} existing rows...")
        
        # Combine new data with existing data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates based on date and ticker, keeping the latest entry
        combined_df = combined_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
        combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        print(f"After merge and deduplication: {len(combined_df)} rows")
        final_df = combined_df
    else:
        print(f"No existing data to merge. Using {len(new_df)} new rows only.")
        final_df = new_df
    
    # Convert date to datetime for Spark compatibility
    # For dates with session markers like "2026-05-30(1)", keep them as strings
    # Spark will store them as strings in the parquet file
    
    # Write merged data back to Volume
    final_spark_df = spark.createDataFrame(final_df)
    final_spark_df.write.mode("overwrite").parquet(raw_path)
    
    print(f"Data saved successfully to Volume: {raw_path}")
    print(f"Total rows in dataset: {len(final_df)}")

def ingest_data():
    """Main ingestion function"""
    # Load configuration
    config = load_config()
    tickers = config['tickers']
    years = config['params']['timeframe_years']
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("StockIngestion") \
        .getOrCreate()
    
    raw_path = "/Volumes/workspace/default/stock_data/raw/stock_data.parquet"
    
    # Check existing data
    file_exists, last_date, existing_df = check_existing_data(spark, raw_path)
    
    # Determine fetch strategy
    fetch_history, fetch_intraday, history_start, history_end = determine_fetch_strategy(
        file_exists, last_date, years
    )
    
    # Load API key (if available, it will be set as env var for vnstock)
    api_key = load_api_key()
    if api_key:
        print(f"Vnstock will use API key (rate limit: 60-180 requests/min)")
    else:
        print(f"Vnstock will run in guest mode (rate limit: 20 requests/min)")
    
    # Get current trading session ONCE (only needed if fetching intraday)
    session_marker = None
    if fetch_intraday:
        session_marker = get_trading_session()[0]  # Only get the session marker
        print(f"Current trading session: {session_marker}")
    
    # Collect all data
    all_history_data = []
    all_intraday_data = []
    
    print(f"\nStarting ingestion for {len(tickers)} tickers...")
    
    for ticker in tickers:
        # Fetch historical data if needed
        if fetch_history:
            df_history = fetch_historical_data(ticker, history_start, history_end)
            if df_history is not None:
                all_history_data.append(df_history)
            time.sleep(0.5)  # Rate limit handling
        
        # Fetch intraday data if needed
        if fetch_intraday:
            df_intraday = fetch_intraday_data(ticker, session_marker)
            if df_intraday is not None:
                all_intraday_data.append(df_intraday)
            time.sleep(0)  # Rate limit handling
    
    # Combine all fetched data
    all_data = []
    if all_history_data:
        all_data.extend(all_history_data)
        print(f"\nFetched historical data: {sum(len(df) for df in all_history_data)} rows")
    if all_intraday_data:
        all_data.extend(all_intraday_data)
        print(f"Fetched intraday data: {sum(len(df) for df in all_intraday_data)} rows")
    
    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal new data fetched: {len(new_df)} rows")
        
        # Merge and save
        merge_and_save_data(spark, raw_path, new_df, existing_df)
    else:
        print("\nNo new data fetched.")

if __name__ == "__main__":
    ingest_data()
