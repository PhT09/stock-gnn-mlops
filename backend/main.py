from fastapi import FastAPI
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from routes import stocks, predictions
from sync import sync_databricks_to_sqlite
from database import check_databricks_connection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup background sync
    scheduler = BackgroundScheduler()
    # Run once at startup
    scheduler.add_job(sync_databricks_to_sqlite)
    # Then run every 12 hours
    scheduler.add_job(sync_databricks_to_sqlite, 'interval', hours=12)
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(title="Stock Predictor API", version="1.0.0", lifespan=lifespan)

app.include_router(stocks.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to Stock Predictor API"}

@app.get("/api/v1/health/databricks")
def health_check_databricks():
    """Endpoint để kiểm tra nhanh xem đã kết nối được lên Databricks chưa"""
    return check_databricks_connection()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)