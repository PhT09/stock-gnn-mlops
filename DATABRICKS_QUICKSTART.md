
================================================================================
📖 HƯỚNG DẪN CHẠY PIPELINE TRÊN DATABRICKS
================================================================================

✅ Đã tạo 2 files mới:
  1. data_engineering/ingestion_databricks.py  → Đọc data từ Volume
  2. mlops/pipeline_databricks.py              → Pipeline cho Databricks

📍 Notebook hướng dẫn: Stock MLOps Pipeline - Databricks


🚀 CÁCH 1: CHẠY TRONG NOTEBOOK (Khuyến nghị để test)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ MỞ NOTEBOOK MỚI TẠO:
   "Stock MLOps Pipeline - Databricks"

2️⃣ HOẶC CHẠY TRỰC TIẾP TẠI ĐÂY:

```python
import sys
sys.path.append('/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops')

from mlops.pipeline_databricks import run_pipeline_databricks

# Chạy pipeline (lần đầu - sẽ train)
result = run_pipeline_databricks()
```

3️⃣ CHẠY LẦN 2 (Sẽ skip vì không có data mới):

```python
result = run_pipeline_databricks()
# Output: ⏸️ PIPELINE PAUSED: Không có data mới
```

4️⃣ BẮT BUỘC TRAIN (Force mode):

```python
result = run_pipeline_databricks(force_train=True)
# Sẽ train dù data không đổi
```

5️⃣ TRAIN CHỈ TRÊN 30 NGÀY GẦN NHẤT:

```python
result = run_pipeline_databricks(recent_days=30)
# Chỉ dùng 30 ngày data mới nhất
```


🔧 CÁCH 2: CHẠY VỚI DATABRICKS JOB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ CẬP NHẬT JOB HIỆN TẠI:

   Job: "New Job May 04, 2026, 01:59 PM"
   Task: stocks_databricks
   
   Thay đổi:
   - File: mlops/pipeline.py
   → Thành: mlops/pipeline_databricks.py


2️⃣ TẠO JOB MỚI (Tùy chọn):

   a. Tạo job mới trong Workflows
   b. Task type: Python file
   c. File path: /Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/mlops/pipeline_databricks.py
   d. Cluster: Serverless (đã có xgboost, scikit-learn)
   e. Schedule: Daily at 5:00 PM (sau khi scraping xong)


3️⃣ PARAMETERS (Optional):

   Trong job settings, thêm parameters:
   ```
   force_train: false
   recent_days: null
   ```


📊 WORKFLOW HOÀN CHỈNH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Scraping Job (Daily 4:00 PM)
   └─ Cào data chứng khoán → Lưu vào Volume
   
2. MLOps Pipeline (Daily 5:00 PM) ← JOB MỚI
   ├─ Ingestion: Đọc từ Volume, check cache
   ├─ Decision: 
   │  ├─ Có data mới → Train model
   │  └─ Không mới → Skip (tiết kiệm compute)
   ├─ Training: XGBoost + MLflow tracking
   └─ Reporting: Generate report + Email


🧪 TEST TỪNG BƯỚC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Chạy từng command để test:

```python
# TEST 1: Chỉ ingestion
from data_engineering.ingestion_databricks import ingest_data
metadata = ingest_data()
print(metadata)

# TEST 2: Check cache
import json
with open('/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/data/.last_processed_date.json') as f:
    print(json.load(f))

# TEST 3: Full pipeline
from mlops.pipeline_databricks import run_pipeline_databricks
result = run_pipeline_databricks()

# TEST 4: Xem MLflow experiment
# Navigate to: /ml/experiments → /Shared/Stock_Prediction_XGBoost
```


⚠️ LƯU Ý QUAN TRỌNG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ Code MỚI (ingestion_databricks.py) đọc trực tiếp từ Volume
   → Nhanh hơn, không cần download

2. ✅ Cache file lưu trong Workspace:
   /Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/data/.last_processed_date.json
   → Persistent giữa các lần chạy job

3. ⚠️  Data hiện tại mới nhất: 2026-04-15 (20 ngày trước)
   → Pipeline sẽ báo warning
   → Cần check scraping job

4. ✅ Email vẫn hoạt động (nếu có SENDER_EMAIL trong .env)


🔍 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ Lỗi: "No module named 'xgboost'"
→ Job đã có xgboost trong environment (đã fix trước đó)
   Kiểm tra: Job settings → Environment → Dependencies

❓ Pipeline luôn skip training
→ Xóa cache file để reset:
   dbutils.fs.rm("file:/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/data/.last_processed_date.json")
   
❓ Muốn xem cache hiện tại
→ with open('/Workspace/.../data/.last_processed_date.json') as f: print(f.read())

❓ Email không gửi được
→ Kiểm tra .env file có SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL


📚 FILE REFERENCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEW FILES (Databricks):
  /stock-gnn-mlops/
  ├── data_engineering/
  │   └── ingestion_databricks.py    ← Đọc từ Volume (Databricks)
  ├── mlops/
  │   ├── pipeline_databricks.py     ← Pipeline cho Databricks
  │   └── pipeline.py                ← Pipeline cho Local
  └── notebooks/
      └── Stock MLOps Pipeline - Databricks.ipynb

EXISTING FILES (Local):
  ├── data_engineering/
  │   └── ingestion.py               ← Download về Local
  └── mlops/
      └── pipeline.py                ← Pipeline cho Local


✅ READY TO RUN!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bây giờ bạn có thể:
1. Mở notebook "Stock MLOps Pipeline - Databricks"
2. Chạy từng cell để test
3. Hoặc chạy trực tiếp job với file pipeline_databricks.py

================================================================================
