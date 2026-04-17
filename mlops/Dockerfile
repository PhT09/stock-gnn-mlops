# Sử dụng Python bản nhẹ (slim) để giảm dung lượng image
FROM python:3.10-slim

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir mlflow[extras] psycopg2-binary boto3

# Thiết lập thư mục làm việc trong container
WORKDIR /mlflow

# Expose port mặc định của MLflow
EXPOSE 5000

# Lệnh khởi chạy server (Sẽ được docker-compose ghi đè nếu cần)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
