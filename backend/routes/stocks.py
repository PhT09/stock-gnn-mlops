from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from database import get_sqlite_db

router = APIRouter(prefix="/stocks", tags=["stocks"])

@router.get("")
def get_stocks(
    query: Optional[str] = Query(None, description="Search ticker containing this string"),
    limit: int = Query(50, description="Max items to return")
):
    try:
        with get_sqlite_db() as cursor:
            if query:
                cursor.execute(
                    "SELECT ticker, close, volume FROM stock_info WHERE ticker LIKE ? LIMIT ?",
                    (f"%{query}%", limit)
                )
            else:
                cursor.execute(
                    "SELECT ticker, close, volume FROM stock_info LIMIT ?",
                    (limit,)
                )
            results = cursor.fetchall()
            data = [dict(row) for row in results]
            
            return {
                "status": "success",
                "count": len(data),
                "data": data
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error (Have you synced Databricks yet?): {e}")


@router.get("/sorted")
def get_sorted_stocks(
    sort_by: str = Query(..., description="Sort by 'price' or 'volume'"),
    order: str = Query("desc", description="Sort order: 'desc' or 'asc'"),
    limit: int = Query(10, description="Number of items to return"),
    offset: int = Query(0, description="Number of items to skip")
):
    if sort_by not in ["price", "volume"]:
        raise HTTPException(status_code=400, detail="Invalid sort_by parameter. Use 'price' or 'volume'.")
    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Invalid order parameter. Use 'asc' or 'desc'.")

    sort_column = "close" if sort_by == "price" else "volume"
    
    try:
        with get_sqlite_db() as cursor:
            cursor.execute("SELECT COUNT(*) FROM stock_info")
            total = cursor.fetchone()[0]
            
            query_sql = f"SELECT ticker, close, volume FROM stock_info ORDER BY {sort_column} {order.upper()} LIMIT ? OFFSET ?"
            cursor.execute(query_sql, (limit, offset))
            results = cursor.fetchall()
            data = [dict(row) for row in results]
            
            return {
                "status": "success",
                "sort_by": sort_by,
                "order": order,
                "total": total,
                "data": data
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")