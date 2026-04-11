from fastapi import FastAPI

app = FastAPI(title="Stock GNN MLOps API")

@app.get("/")
def read_root():
    return {"message": "Welcome to Stock GNN MLOps API"}

@app.get("/data")
def get_data():
    return {"message": "Data endpoint: Fetch and process stock data here."}

@app.get("/graph")
def get_graph():
    return {"message": "Graph endpoint: Construct and analyze stock graphs here."}

@app.get("/predict")
def get_predict():
    return {"message": "Predict endpoint: Run GNN models for stock predictions here."}
