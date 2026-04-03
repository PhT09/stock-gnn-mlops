# SCALABLE CLOUD-BASED MLOPS PIPELINE FOR STOCK TREND PREDICTION USING GRAPH MINING AND GRAPH NEURAL NETWORKS


## Team Members

| Họ và tên               | MSSV     |
|------------------------|----------|
| Võ Đại Phát           | 23672291 |
| Trần Hoàng Xuân Lộc   | 23636491 |
| Phạm Ngọc Toàn        | 23672111 |
| Trần Anh Kiệt         | 23655711 |
| Trần Nguyễn Toàn Phát | 23643121 |

---

## Overview
This project builds a scalable **MLOps pipeline** for stock trend prediction using:

- Apache Spark (data processing)  
- Graph Mining (NetworkX, Node2Vec)  
- GNN (GraphSAGE / GCN)  
- MLflow (tracking)  
- FastAPI + Docker (deployment)  

---

## Pipeline
1. **Data Ingestion** (vnstock)  
2. **Spark Processing** (feature engineering)  
3. **Graph Construction** (correlation → graph)  
4. **GNN Training**  
5. **Deployment** (FastAPI microservices)  

---

## Team Structure
- **A:** Data Engineering  
- **B:** Graph Mining  
- **C:** Machine Learning  
- **D:** MLOps  
- **E:** Backend  

---

## Run Project
```bash
pip install -r requirements.txt
```

---

## Run Docker
```bash
docker-compose up --build
```

