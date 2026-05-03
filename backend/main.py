from fastapi import FastAPI
from backend.routes import data, predict

app = FastAPI(title="Stock Predictor API", version="1.0.0")

app.include_router(data.router)
app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "Welcome to Stock Predictor API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
