import os
from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException
from backend.database import get_sqlite_db

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Get table name from environment, default to 'stock_predictions'
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


@router.get("/status")
def get_sync_status():
    """5. Kiểm tra trạng thái đồng bộ dữ liệu (Metadata & Sync Status)"""
    query = f"""
        SELECT 
            MAX(latest_data_date) as latest_data_date,
            COUNT(DISTINCT ticker) as total_predicted_tickers
        FROM {TABLE_NAME}
    """
    data = execute_query(query)
    
    if not data or not data[0].get("latest_data_date"):
        return {
            "latest_data_date": None,
            "total_predicted_tickers": 0,
            "status": "UNHEALTHY"
        }
        
    result = data[0]
    return {
        "latest_data_date": result["latest_data_date"],
        "total_predicted_tickers": result["total_predicted_tickers"],
        "status": "HEALTHY"
    }


@router.get("/recommendations")
def get_recommendations(limit: int = Query(5, description="Number of recommendations")):
    """4. Gợi ý các cổ phiếu tiềm năng nhất (Top Recommendations)"""
    query = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE latest_data_date = (SELECT MAX(latest_data_date) FROM {TABLE_NAME})
          AND day_1_signal = 'BUY'
          AND day_2_signal = 'BUY'
          AND day_3_signal = 'BUY'
    """
    data = execute_query(query)
    
    recommendations = []
    for row in data:
        # Calculate average probability for the first 3 days
        probs = [
            float(row.get("day_1_probability", 0)),
            float(row.get("day_2_probability", 0)),
            float(row.get("day_3_probability", 0))
        ]
        avg_prob = sum(probs) / len(probs)
        
        recommendations.append({
            "ticker": row["ticker"],
            "signals": ["BUY", "BUY", "BUY"],
            "average_probability": round(avg_prob, 3)
        })
        
    # Sort by average probability descending
    recommendations.sort(key=lambda x: x["average_probability"], reverse=True)
    
    return {
        "recommendation_criteria": "Tín hiệu MUA liên tiếp trong 3 phiên giao dịch đầu tiên",
        "data": recommendations[:limit]
    }


@router.get("/next")
def get_next_session_batch(tickers: Optional[str] = Query(None, description="Comma-separated list of tickers")):
    """2. Dự đoán phiên giao dịch tiếp theo hàng loạt (Watchlist / Toàn thị trường)"""
    where_clause = f"latest_data_date = (SELECT MAX(latest_data_date) FROM {TABLE_NAME})"
    params = None
    
    # If tickers provided, filter by them. Note: databricks sql connector might not support array parameters perfectly
    # so we carefully format the IN clause if tickers are provided, or use parameters if supported.
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        placeholders = ",".join(["?"] * len(ticker_list))
        where_clause += f" AND ticker IN ({placeholders})"
        params = tuple(ticker_list)
        
    query = f"""
        SELECT 
            ticker, latest_data_date, day_1_date, day_1_prediction, day_1_signal, day_1_probability, day_1_confidence
        FROM {TABLE_NAME}
        WHERE {where_clause}
    """
    
    data = execute_query(query, params)
    
    if not data:
        return {"predictions": []}
        
    latest_date = data[0].get("latest_data_date")
    next_date = data[0].get("day_1_date")
    
    predictions = []
    for row in data:
        predictions.append({
            "ticker": row["ticker"],
            "prediction": row["day_1_prediction"],
            "signal": row["day_1_signal"],
            "probability": row["day_1_probability"],
            "confidence": row["day_1_confidence"]
        })
        
    return {
        "calculated_date": latest_date,
        "next_session_date": next_date,
        "predictions": predictions
    }


@router.get("/{ticker}/next")
def get_next_session_single(ticker: str):
    """1. Dự đoán phiên giao dịch tiếp theo cho 1 cổ phiếu"""
    query = f"""
        SELECT 
            ticker, latest_data_date, day_1_date, day_1_prediction, day_1_signal, day_1_probability, day_1_confidence
        FROM {TABLE_NAME}
        WHERE ticker = ?
        ORDER BY latest_data_date DESC
        LIMIT 1
    """
    data = execute_query(query, (ticker.upper(),))
    
    if not data:
        raise HTTPException(status_code=404, detail=f"No prediction found for {ticker}")
        
    row = data[0]
    return {
        "ticker": row["ticker"],
        "calculated_date": row["latest_data_date"],
        "next_session": {
            "date": row["day_1_date"],
            "prediction": row["day_1_prediction"],
            "signal": row["day_1_signal"],
            "probability": row["day_1_probability"],
            "confidence": row["day_1_confidence"]
        }
    }


@router.get("/{ticker}")
def get_15_days_forecast(ticker: str):
    """3. Dự báo chuỗi 15 ngày tiếp theo của 1 cổ phiếu (Phục vụ vẽ biểu đồ)"""
    query = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE ticker = ?
        ORDER BY latest_data_date DESC
        LIMIT 1
    """
    data = execute_query(query, (ticker.upper(),))
    
    if not data:
        raise HTTPException(status_code=404, detail=f"No prediction found for {ticker}")
        
    row = data[0]
    
    forecast = []
    for day in range(1, 16):
        date_key = f"day_{day}_date"
        if row.get(date_key):
            forecast.append({
                "day": day,
                "date": row.get(f"day_{day}_date"),
                "prediction": row.get(f"day_{day}_prediction"),
                "signal": row.get(f"day_{day}_signal"),
                "probability": row.get(f"day_{day}_probability"),
                "confidence": row.get(f"day_{day}_confidence")
            })
            
    return {
        "ticker": row["ticker"],
        "latest_data_date": row["latest_data_date"],
        "forecast": forecast
    }
