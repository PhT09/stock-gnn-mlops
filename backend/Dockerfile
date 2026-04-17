# Sử dụng Python 3.11-slim để đảm bảo độ ổn định cao nhất cho các thư viện ML
FROM python:3.11-slim

WORKDIR /app

# Cài đặt các công cụ hệ thống cần thiết để biên dịch (nếu cần)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Nâng cấp pip và cài đặt thư viện
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
ENV PYTHONPATH=/app

CMD ["python", "backend/main.py"]
