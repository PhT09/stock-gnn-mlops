import os
import yaml
import pandas as pd
from datetime import datetime, timedelta, time as dt_time, timezone
import time
import logging
from pyspark.sql import SparkSession
from dotenv import load_dotenv

from vnstock.api.quote import Quote
from tenacity import retry_if_exception_type, retry_unless_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure vnstock Quote to not retry on ValueError (empty data) to save time & API budget
# MUST combine with retry_if_exception_type(Exception) so it doesn't retry on successful results!
retry_cond = retry_if_exception_type(Exception) & retry_unless_exception_type(ValueError)
Quote.history.retry.retry = retry_cond
Quote.intraday.retry.retry = retry_cond

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
            logger.info("VNSTOCK API key loaded from .env file")
            os.environ['VNSTOCK_API_KEY'] = api_key
            return api_key
        else:
            logger.warning("VNSTOCK_API not found in .env file. Using guest mode.")
            return api_key
    else:
        logger.warning(f".env file not found at {env_path}. Using guest mode.")
        return None

def get_trading_session():
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
    morning_end = datetime.strptime("11:30", "%H:%M").time()
    afternoon_start = datetime.strptime("13:30", "%H:%M").time()
    afternoon_end = datetime.strptime("15:00", "%H:%M").time()
    
    if morning_start <= current_time < afternoon_start:
        return '(1)', current_time, morning_start, afternoon_start
    elif afternoon_start <= current_time:
        return '(2)', current_time, afternoon_start
    else:
        return '(2)', current_time, morning_start, morning_end, afternoon_start, afternoon_end

def check_existing_data(spark, raw_path):
    """Check if data file exists and return last date if available"""
    try:
        existing_spark_df = spark.read.parquet(raw_path)
        existing_df = existing_spark_df.toPandas()
        
        if not existing_df.empty:
            existing_df['date_clean'] = existing_df['date'].astype(str).str.replace(r'\(\d+\)', '', regex=True)
            existing_df['date_parsed'] = pd.to_datetime(existing_df['date_clean'])
            max_date = existing_df['date_parsed'].max().date()
            existing_df = existing_df.drop(columns=['date_clean', 'date_parsed'])
            
            logger.info(f"Existing data found with {len(existing_df):,} rows. Last date: {max_date}")
            return True, max_date, existing_df
        else:
            logger.info("Existing data file found but empty.")
            return True, None, existing_df
            
    except Exception as e:
        logger.info(f"No existing data found ({str(e)}). Will create new file.")
        return False, None, None

def determine_fetch_strategy(file_exists, last_date, years):
    """Determine what data to fetch based on existing data"""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    if not file_exists or last_date is None:
        history_start = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        logger.info(f"Strategy: Full load - history from {history_start} to {yesterday_str} + today's intraday")
        return True, True, history_start, yesterday_str
    
    if last_date == yesterday:
        logger.info("Strategy: Data up to date through yesterday. Fetching only today's intraday data")
        return False, True, None, None
    
    elif last_date < yesterday:
        history_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        logger.info(f"Strategy: Gap detected. Fetching history from {history_start} to {yesterday_str} + today's intraday")
        return True, True, history_start, yesterday_str
    
    else:
        logger.info("Strategy: Data includes today. Fetching today's intraday to update")
        return False, True, None, None

def fetch_historical_data(ticker, start_date, end_date, prefix=""):
    """Fetch historical data for a ticker using Quote.history()"""
    p_str = prefix if prefix else f"  {ticker}"
    try:
        quote = Quote(symbol=ticker, source='KBS')
        df_history = quote.history(start=start_date, end=end_date)
        
        if df_history is not None and not df_history.empty:
            df_history = df_history.rename(columns={'time': 'date'})
            df_history['ticker'] = ticker
            df_history = df_history[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
            df_history['date'] = pd.to_datetime(df_history['date']).dt.strftime('%Y-%m-%d')

            logger.info(f"{p_str} - History: {len(df_history)} rows from {start_date} to {end_date}")
            return df_history
        else:
            logger.warning(f"{p_str} - No historical data")
            return None
            
    except (Exception, SystemExit) as e:
        logger.error(f"{p_str} - Error fetching history: {str(e)}")
        if "rate limit" in str(e).lower() or "systemexit" in str(type(e).__name__).lower():
            logger.warning("Rate limit hit. Sleeping for 60 seconds before continuing...")
            time.sleep(60)
        return None

def fetch_intraday_data(ticker, session_marker, prefix=""):
    """Fetch today's intraday data for a ticker using Quote.intraday()"""
    p_str = prefix if prefix else f"  {ticker}"
    try:
        quote = Quote(symbol=ticker, source='KBS')
        df_intraday = quote.intraday(page_size=10000)
        
        if df_intraday is not None and not df_intraday.empty:
            df_intraday = df_intraday.rename(columns={'time': 'date'})
            df_intraday['datetime'] = pd.to_datetime(df_intraday['date'])
            df_intraday['time'] = df_intraday['datetime'].dt.time
            
            if session_marker == '(1)':
                session_start = dt_time(9, 0)
                session_end = dt_time(11, 30)
            else:
                session_start = dt_time(13, 30)
                session_end = dt_time(15, 0)
            
            df_session = df_intraday[
                (df_intraday['time'] >= session_start) & 
                (df_intraday['time'] <= session_end)
            ].copy()
            
            if df_session.empty:
                logger.warning(f"{p_str} - No intraday data for session {session_marker}")
                return None
            
            session_date = df_session['datetime'].dt.date.iloc[0]
            
            ohlcv_data = {
                'date': f"{session_date}{session_marker}",
                'ticker': ticker,
                'open': df_session['price'].iloc[0],
                'high': df_session['price'].max(),
                'low': df_session['price'].min(),
                'close': df_session['price'].iloc[-1],
                'volume': df_session['volume'].sum()
            }
            
            result_df = pd.DataFrame([ohlcv_data])
            logger.info(f"{p_str} - Intraday: session {session_marker} ({len(df_session)} ticks)")
            return result_df
        else:
            logger.warning(f"{p_str} - No intraday data")
            return None
            
    except (Exception, SystemExit) as e:
        logger.error(f"{p_str} - Error fetching intraday: {str(e)}")
        if "rate limit" in str(e).lower() or "systemexit" in str(type(e).__name__).lower():
            logger.warning("Rate limit hit. Sleeping for 60 seconds before continuing...")
            time.sleep(60)
        return None

def merge_and_save_data(spark, raw_path, new_df, existing_df):
    """Merge new data with existing data and save to parquet"""
    if existing_df is not None and not existing_df.empty:
        logger.info(f"Merging {len(new_df):,} new rows with {len(existing_df):,} existing rows...")
        
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
        combined_df = combined_df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        logger.info(f"After merge and deduplication: {len(combined_df):,} rows")
        final_df = combined_df
    else:
        logger.info(f"No existing data to merge. Using {len(new_df):,} new rows only.")
        final_df = new_df
    
    # Ensure date column is explicitly string type for Spark compatibility
    final_df['date'] = final_df['date'].astype(str)
    
    # Write merged data back to Volume
    final_spark_df = spark.createDataFrame(final_df)
    final_spark_df.write.mode("overwrite").parquet(raw_path)
    
    logger.info(f"Data saved successfully to Volume: {raw_path}")
    logger.info(f"Total rows in dataset: {len(final_df):,}")

def ingest_data():
    """Main ingestion function"""
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("STOCK DATA INGESTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
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
    
    # Load API key
    api_key = load_api_key()
    if api_key:
        logger.info("Vnstock will use API key (rate limit: 60-180 requests/min)")
        delay = 0.5
    else:
        logger.info("Vnstock will run in guest mode (rate limit: 20 requests/min)")
        delay = 3.0
    
    # Get current trading session
    session_marker = None
    if fetch_intraday:
        session_marker = get_trading_session()[0]
        logger.info(f"Current trading session: {session_marker}")
    
    # Collect all data
    all_history_data = []
    all_intraday_data = []
    
    total_tickers = len(tickers)
    logger.info(f"\nStarting ingestion for {total_tickers} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        prefix = f"[{i}/{total_tickers}] {ticker}"
        
        # Fetch historical data if needed
        if fetch_history:
            df_history = fetch_historical_data(ticker, history_start, history_end, prefix=prefix)
            if df_history is not None:
                all_history_data.append(df_history)
            time.sleep(delay)
        
        # Fetch intraday data if needed
        if fetch_intraday:
            df_intraday = fetch_intraday_data(ticker, session_marker, prefix=prefix)
            if df_intraday is not None:
                all_intraday_data.append(df_intraday)
            time.sleep(delay)
    
    # Combine all fetched data
    all_data = []
    if all_history_data:
        all_data.extend(all_history_data)
        total_history = sum(len(df) for df in all_history_data)
        logger.info(f"\nFetched historical data: {total_history:,} rows")
    if all_intraday_data:
        all_data.extend(all_intraday_data)
        total_intraday = sum(len(df) for df in all_intraday_data)
        logger.info(f"Fetched intraday data: {total_intraday:,} rows")
    
    if all_data:
        new_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"\nTotal new data fetched: {len(new_df):,} rows")
        
        # Merge and save
        merge_and_save_data(spark, raw_path, new_df, existing_df)
    else:
        logger.warning("\nNo new data fetched.")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*80)
    logger.info("INGESTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Start time:        {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time:          {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration:          {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info("="*80)
    logger.info("INGESTION COMPLETED SUCCESSFULLY ✓")
    logger.info("="*80 + "\n")

if __name__ == "__main__":
    ingest_data()