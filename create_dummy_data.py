import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

# Create data directory
os.makedirs("data", exist_ok=True)

# 1. Create dummy CSV for Stocks API
csv_data = [
    {"ticker": "FPT", "close_price": 135200.0, "volume": 2500000},
    {"ticker": "VCB", "close_price": 92100.0, "volume": 1200000},
    {"ticker": "HPG", "close_price": 28400.0, "volume": 15400000},
    {"ticker": "SSI", "close_price": 36500.0, "volume": 8500000},
    {"ticker": "VNM", "close_price": 67000.0, "volume": 3200000},
]
df_csv = pd.DataFrame(csv_data)
df_csv.to_csv("data/ticker_price_volume.csv", index=False)
print("Created dummy data/ticker_price_volume.csv")

# 2. Create dummy SQLite DB for Predictions API
db_path = "data/predictions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS stock_predictions (
    ticker TEXT,
    latest_data_date TEXT,
    day_1_date TEXT, day_1_prediction INTEGER, day_1_signal TEXT, day_1_probability REAL, day_1_confidence TEXT,
    day_2_date TEXT, day_2_prediction INTEGER, day_2_signal TEXT, day_2_probability REAL, day_2_confidence TEXT,
    day_3_date TEXT, day_3_prediction INTEGER, day_3_signal TEXT, day_3_probability REAL, day_3_confidence TEXT,
    day_4_date TEXT, day_4_prediction INTEGER, day_4_signal TEXT, day_4_probability REAL, day_4_confidence TEXT,
    day_5_date TEXT, day_5_prediction INTEGER, day_5_signal TEXT, day_5_probability REAL, day_5_confidence TEXT,
    day_6_date TEXT, day_6_prediction INTEGER, day_6_signal TEXT, day_6_probability REAL, day_6_confidence TEXT,
    day_7_date TEXT, day_7_prediction INTEGER, day_7_signal TEXT, day_7_probability REAL, day_7_confidence TEXT,
    day_8_date TEXT, day_8_prediction INTEGER, day_8_signal TEXT, day_8_probability REAL, day_8_confidence TEXT,
    day_9_date TEXT, day_9_prediction INTEGER, day_9_signal TEXT, day_9_probability REAL, day_9_confidence TEXT,
    day_10_date TEXT, day_10_prediction INTEGER, day_10_signal TEXT, day_10_probability REAL, day_10_confidence TEXT,
    day_11_date TEXT, day_11_prediction INTEGER, day_11_signal TEXT, day_11_probability REAL, day_11_confidence TEXT,
    day_12_date TEXT, day_12_prediction INTEGER, day_12_signal TEXT, day_12_probability REAL, day_12_confidence TEXT,
    day_13_date TEXT, day_13_prediction INTEGER, day_13_signal TEXT, day_13_probability REAL, day_13_confidence TEXT,
    day_14_date TEXT, day_14_prediction INTEGER, day_14_signal TEXT, day_14_probability REAL, day_14_confidence TEXT,
    day_15_date TEXT, day_15_prediction INTEGER, day_15_signal TEXT, day_15_probability REAL, day_15_confidence TEXT
)
""")
cursor.execute("DELETE FROM stock_predictions")

today = datetime.now().strftime("%Y-%m-%d")
dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 16)]

# Dummy data generation helper
def generate_row(ticker, signals):
    row = [ticker, today]
    for i in range(15):
        signal = signals[i] if i < len(signals) else "SELL"
        pred = 1 if signal == "BUY" else 0
        prob = 0.75 if signal == "BUY" else 0.35
        conf = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.5 else "LOW")
        row.extend([dates[i], pred, signal, prob, conf])
    return row

# Insert FPT (3 consecutive BUYs for recommendation), HPG (also 3 BUYs), VCB (SELL)
rows = [
    generate_row("FPT", ["BUY", "BUY", "BUY", "SELL", "BUY"]),
    generate_row("HPG", ["BUY", "BUY", "BUY", "BUY", "BUY"]),
    generate_row("VCB", ["SELL", "BUY", "SELL", "SELL", "SELL"])
]

placeholders = ",".join(["?"] * 77) # 2 cols + 15*5 cols
cursor.executemany(f"INSERT INTO stock_predictions VALUES ({placeholders})", rows)
conn.commit()
conn.close()

print("Created dummy data/predictions.db")
