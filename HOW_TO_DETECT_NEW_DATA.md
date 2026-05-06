
================================================================================
📖 HƯỚNG DẪN: PHÂN BIỆT & XỬ LÝ DATA MỚI/CŨ
================================================================================

❓ CÂU HỎI: "Làm sao phân biệt file data mới và cũ khi tải về?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔴 TRẢ LỜI NGẮN: KHÔNG THỂ dựa vào tên file hay modification time

✅ CÁCH DUY NHẤT: Đọc cột 'date' trong data và so sánh với lần trước


🏗️ CẤU TRÚC DỮ LIỆU DATABRICKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Path: /Volumes/workspace/default/stock_data/processed/stock_features.parquet/

📁 stock_features.parquet/
   ├── part-00000.parquet  (7.5 MB)
   ├── part-00001.parquet  (7.2 MB)
   ├── ...
   ├── part-00010.parquet  (3.8 MB)
   ├── _committed_XXX (transaction logs)
   └── _started_XXX

❌ Các part file KHÔNG PHÂN PARTITION theo ngày
❌ Tên file KHÔNG chứa thông tin ngày cào
✅ Mỗi file chứa data của NHIỀU ngày giao dịch
✅ Modification time = Lần ghi cuối (2026-04-16 17:13:29)


📊 DATA SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Column          Type           Ý nghĩa
-----------     -----------    ------------------
date            timestamp      Ngày GIAO DỊCH chứng khoán (KHÔNG phải ngày cào)
ticker          string         Mã cổ phiếu
scaled_features vectorudt      Features đã chuẩn hóa
target          integer        Label (0=giảm, 1=tăng)

⚠️ KHÔNG có: scraped_at, ingested_at, version, batch_id


💡 GIẢI PHÁP: CACHE MECHANISM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

File cache: data/.last_processed_date.json

{
  "latest_date": "2026-04-16",
  "processed_at": "2026-05-05T10:30:00"
}

WORKFLOW:

1. Download data từ Databricks
2. Đọc max(date) → "2026-04-16"
3. Đọc cache → "2026-04-15"
4. So sánh:
   - 2026-04-16 > 2026-04-15 → ✅ CÓ DATA MỚI → Train model
   - 2026-04-16 = 2026-04-15 → ⏸️ KHÔNG CÓ MỚI → Skip
5. Lưu cache: "2026-04-16"


🚀 SỬ DỤNG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ CHẠY PIPELINE THÔNG THƯỜNG (Tự động skip nếu không có data mới):

   python mlops/pipeline.py
   
   Output:
   ⏸️ PIPELINE PAUSED: Không có data mới
      Lần xử lý cuối: 2026-05-05T10:30:00
      Dùng run_pipeline(force_train=True) để bắt buộc train


2️⃣ BẮT BUỘC TRAIN (Dù data không đổi):

   # Trong code:
   run_pipeline(force_train=True)
   
   # Hoặc sửa pipeline.py để thêm argument


3️⃣ CHECK DATA METADATA (Không train):

   from data_engineering.ingestion import ingest_data
   
   metadata = ingest_data()
   print(f"Has new data: {metadata['has_new_data']}")
   print(f"Latest date: {metadata['latest_date']}")
   print(f"Last processed: {metadata['last_processed_date']}")


📋 KẾT LUẬN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ KHÔNG CẦN partition by scraping date
✅ KHÔNG CẦN đổi tên file theo ngày
✅ Dùng CACHE để track data mới
✅ Pipeline tự động skip khi không có data mới
✅ Tiết kiệm compute & MLflow storage

⚠️ LƯU Ý: Data hiện tại mới nhất là 2026-04-16 (19 ngày trước)
          → Cần kiểm tra scraping job


🔮 NÂNG CAO (TƯƠNG LAI)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Chuyển sang Delta Table (Time travel, ACID transactions)
2. Thêm scraped_at column vào scraping job
3. Incremental training (chỉ train trên data mới, không retrain tất cả)
4. Model versioning strategy (khi nào deploy model mới)

================================================================================
