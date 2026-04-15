import os
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Stock GNN MLOps API")

@app.get("/")
def read_root():
    return {"message": "Welcome to Stock GNN MLOps API"}

@app.get("/data")
def get_data(limit: int = 100):
    try:
        import os
        import glob
        import json
        
        data_path = "downloaded_data"
        
        # Kiểm tra xem folder data đã tồn tại chưa
        if not os.path.exists(data_path):
            raise HTTPException(
                status_code=404, 
                detail="Chưa có dữ liệu. Vui lòng chạy script download_databricks.py trước."
            )
            
        # Chỉ giới hạn đọc các file CÓ ĐUÔI .parquet (Bỏ qua các file rác _started, _committed của Spark)
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        
        if len(parquet_files) == 0:
            raise HTTPException(
                status_code=404, 
                detail="Trong folder data chưa có file đuôi .parquet nào."
            )
            
        # Đọc và ghép nối tất cả các file parquet
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        
        # Giới hạn số dòng trả về để tránh nặng API (mặc định lấy 100 dòng)
        df_subset = df.head(limit)
        
        # CÁCH CHUẨN NHẤT: Dùng df.to_json() sẽ tự động convert mọi NaT, NaN, Timestamp thành giá trị JSON tương thích tuyệt đối. 
        # Sau đó loads trở lại thành dict để FastAPI trả về
        json_string = df_subset.to_json(orient="records", date_format="iso")
        records = json.loads(json_string)
        
        return {
            "total_records": len(df),
            "returned_records": len(records),
            "limit": limit,
            "data": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc dữ liệu Parquet: {str(e)}")

@app.get("/graph")
def get_graph():
    return {"message": "Graph endpoint: Construct and analyze stock graphs here."}

@app.get("/predict")
def get_predict():
    return {"message": "Predict endpoint: Run GNN models for stock predictions here."}
