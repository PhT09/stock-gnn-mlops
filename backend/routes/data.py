from fastapi import APIRouter
from data_engineering.ingestion import ingest_data

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/stocks")
def get_stocks_info():
    return {"message": "Stock data available in data/processed"}

@router.post("/update")
def update_stock_data():
    ingest_data()
    return {"message": "Data ingestion triggered from Databricks"}
