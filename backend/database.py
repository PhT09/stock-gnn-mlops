import os
from contextlib import contextmanager
import sqlite3
from databricks import sql
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

DATABRICKS_SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

@contextmanager
def get_databricks_db():
    """
    Context manager cung cấp connection/cursor truy vấn Databricks SQL.
    Tự động đóng kết nối sau khi hoàn thành công việc.
    """
    connection = sql.connect(
        server_hostname=DATABRICKS_SERVER_HOSTNAME,
        http_path=DATABRICKS_HTTP_PATH,
        access_token=DATABRICKS_TOKEN
    )
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        connection.close()

SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "predictions.db")

@contextmanager
def get_sqlite_db():
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    connection = sqlite3.connect(SQLITE_DB_PATH)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        connection.close()

def check_databricks_connection():
    """
    Hàm kiểm tra kết nối Databricks bằng cách thử thực thi câu lệnh SQL đơn giản.
    """
    try:
        with get_databricks_db() as cursor:
            cursor.execute("SELECT 1")
            return {"status": "success", "message": "Successfully connected to Databricks SQL Warehouse"}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}