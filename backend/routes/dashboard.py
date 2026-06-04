"""
Dashboard endpoints — phục vụ FE hiển thị market overview.
"""
import os
from fastapi import APIRouter, HTTPException
from database import get_sqlite_db

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

TABLE_NAME = os.getenv("DATABRICKS_TABLE_NAME", "stock_predictions")


def execute_query(query: str, parameters: tuple = None):
    try:
        with get_sqlite_db() as cursor:
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")


@router.get("/summary")
def get_market_summary():
    """
    Market overview cho dashboard chính của FE.

    Trả về:
    - total_tickers: tổng số mã được dự đoán
    - next_session_date: ngày phiên kế tiếp
    - up_count / down_count: số mã được dự đoán TĂNG / GIẢM ngày kế tiếp
    - up_percent: % mã được dự đoán TĂNG
    - confidence_breakdown: phân bố HIGH/MEDIUM/LOW confidence
    - high_confidence_count: số mã có dự đoán HIGH confidence
    - market_sentiment: BULLISH (>55% UP), BEARISH (<45% UP), NEUTRAL (45-55%)
    """
    summary_query = f"""
        SELECT
            COUNT(*) AS total_tickers,
            MAX(day_1_date) AS next_session_date,
            SUM(CASE WHEN day_1_prediction = 1 THEN 1 ELSE 0 END) AS up_count,
            SUM(CASE WHEN day_1_prediction = 0 THEN 1 ELSE 0 END) AS down_count,
            SUM(CASE WHEN day_1_confidence = 'HIGH' THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN day_1_confidence = 'MEDIUM' THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN day_1_confidence = 'LOW' THEN 1 ELSE 0 END) AS low_count,
            AVG(day_1_probability) AS avg_probability
        FROM {TABLE_NAME}
    """
    data = execute_query(summary_query)
    if not data:
        raise HTTPException(status_code=404, detail="No prediction data available")

    row = data[0]
    total = row["total_tickers"] or 0
    up = row["up_count"] or 0
    down = row["down_count"] or 0
    up_pct = round(up / total * 100, 2) if total else 0

    if up_pct >= 55:
        sentiment = "BULLISH"
    elif up_pct <= 45:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    return {
        "total_tickers": total,
        "next_session_date": row["next_session_date"],
        "up_count": up,
        "down_count": down,
        "up_percent": up_pct,
        "down_percent": round(100 - up_pct, 2),
        "market_sentiment": sentiment,
        "high_confidence_count": row["high_count"] or 0,
        "confidence_breakdown": {
            "HIGH": row["high_count"] or 0,
            "MEDIUM": row["medium_count"] or 0,
            "LOW": row["low_count"] or 0,
        },
        "avg_probability": round(row["avg_probability"] or 0, 4),
    }
