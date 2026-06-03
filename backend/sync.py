import os
import pandas as pd
from sqlalchemy import create_engine
from database import get_databricks_db, SQLITE_DB_PATH

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

_table_versions = {}

def get_table_version(cursor, table_name):
    try:
        cursor.execute(f"DESCRIBE HISTORY {table_name} LIMIT 1")
        cols = [desc[0].lower() for desc in cursor.description]
        result = cursor.fetchone()
        if result:
            if 'version' in cols:
                return result[cols.index('version')]
            return result[0]
    except Exception as e:
        print(f"Could not get version for {table_name} (might not be a Delta table): {e}")
    return None

def check_and_sync():
    global _table_versions
    try:
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        abs_path = os.path.abspath(SQLITE_DB_PATH).replace('\\', '/')
        engine = create_engine(f"sqlite:///{abs_path}")

        tables_to_sync = [
            (TABLE_NAME, "stock_predictions"),
            ("workspace.default.ticker_price_volume", "stock_info")
        ]

        with get_databricks_db() as cursor:
            for source_table, target_table in tables_to_sync:
                latest_version = get_table_version(cursor, source_table)
                last_version = _table_versions.get(source_table)
                
                # Compare versions. If latest_version is None, we always sync to be safe.
                if latest_version is None or last_version != latest_version:
                    print(f"Table {source_table} has changed (version {last_version} -> {latest_version}). Syncing...")
                    _sync_single_table(cursor, engine, source_table, target_table)
                    _table_versions[source_table] = latest_version
                else:
                    print(f"Table {source_table} is up to date (version {latest_version}). Skipping sync.")
                    
    except Exception as e:
        print(f"Sync check failed: {e}")

def sync_databricks_to_sqlite():
    check_and_sync()