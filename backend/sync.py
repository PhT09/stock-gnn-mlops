import os
import pandas as pd
from sqlalchemy import create_engine
from backend.database import get_databricks_db, SQLITE_DB_PATH

TABLE_NAME = os.getenv("DATABRICKS_TABLE_NAME", "stock_predictions")

def _sync_single_table(cursor, engine, source_table, target_table):
    print(f"Starting sync from Databricks table {source_table} to local SQLite...")
    cursor.execute(f"SELECT * FROM {source_table}")
    columns = [desc[0] for desc in cursor.description]
    results = cursor.fetchall()
    
    if not results:
        print(f"No data found in Databricks for {source_table}.")
        return
        
    df = pd.DataFrame(results, columns=columns)
    df.to_sql(target_table, con=engine, if_exists="replace", index=False)
    print(f"Successfully synced {len(df)} rows from {source_table} to SQLite.")

def sync_databricks_to_sqlite():
    try:
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        abs_path = os.path.abspath(SQLITE_DB_PATH).replace('\\', '/')
        engine = create_engine(f"sqlite:///{abs_path}")

        with get_databricks_db() as cursor:
            _sync_single_table(cursor, engine, TABLE_NAME, "stock_predictions")
            _sync_single_table(cursor, engine, "workspace.default.ticker_price_volume", "stock_info")
            
    except Exception as e:
        print(f"Sync failed: {e}")
