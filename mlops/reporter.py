import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

def generate_report(model, df, metrics, data_metadata=None):
    print("Đang tạo báo cáo Insight & Visualization...")
    
    # 1. Trích xuất Feature Importances từ XGBoost
    importances = model.feature_importances_
    
    # Gán tên cho 16 features
    feature_names = [
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "price_vs_ma5",
        "price_vs_ma10",
        "ma5_vs_ma10",
        "volume_ratio",
        "volume_change",
        "volatility_5",
        "volatility_10",
        "oc_return",
        "hl_range",
        "close_position",
        "return_lag1",
        "return_lag2",
        "return_lag3"
    ]
    
    # Lấy top 3 features quan trọng nhất
    top_indices = np.argsort(importances)[::-1][:3]
    top_features = [(feature_names[i], importances[i]) for i in top_indices]
    
    # ===== PHẦN MỚI: PREDICT TẤT CẢ CÁC MÃ Ở NGÀY MỚI NHẤT =====
    print("\n📊 Phân tích tất cả các mã cổ phiếu...")
    
    # 2. Lấy ngày mới nhất và tất cả các mã ở ngày đó
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date].copy()
    
    print(f"   • Ngày phân tích: {latest_date}")
    print(f"   • Số lượng mã: {len(df_latest)}")
    
    # 3. Extract features và predict TẤT CẢ
    X_latest = np.vstack(df_latest['scaled_features'].apply(lambda x: x['values']).values)
    predictions = model.predict(X_latest)
    proba = model.predict_proba(X_latest)[:, 1]  # Confidence cho class 1 (TĂNG)
    
    # 4. Thêm predictions vào dataframe
    df_latest['prediction'] = predictions
    df_latest['confidence'] = proba
    df_latest['signal'] = df_latest['prediction'].apply(lambda x: 'TĂNG 📈' if x == 1 else 'GIẢM 📉')
    
    # 5. Tính statistics
    up_count = sum(predictions == 1)
    down_count = sum(predictions == 0)
    total = len(predictions)
    up_pct = (up_count / total) * 100
    down_pct = 100 - up_pct
    
    # 6. Xác định sentiment thị trường
    if up_pct > 60:
        market_sentiment = "XU HƯỚNG TĂNG 📈"
        sentiment_emoji = "📈"
    elif up_pct < 40:
        market_sentiment = "XU HƯỚNG GIẢM 📉"
        sentiment_emoji = "📉"
    else:
        market_sentiment = "SIDEWAY ↔️"
        sentiment_emoji = "↔️"
    
    print(f"\n   📊 Kết quả phân tích:")
    print(f"      • {up_count}/{total} mã TĂNG ({up_pct:.1f}%)")
    print(f"      • {down_count}/{total} mã GIẢM ({down_pct:.1f}%)")
    print(f"      • Thị trường: {market_sentiment}")
    
    # 7. Top stocks
    if up_count > 0:
        top_up = df_latest[df_latest['prediction'] == 1].nlargest(5, 'confidence')[['ticker', 'confidence']]
        print(f"\n   🔥 Top 5 mã TĂNG:")
        for idx, row in top_up.iterrows():
            print(f"      {row['ticker']}: {row['confidence']*100:.1f}%")
    else:
        top_up = pd.DataFrame(columns=['ticker', 'confidence'])
    
    if down_count > 0:
        top_down = df_latest[df_latest['prediction'] == 0].nlargest(5, 'confidence')[['ticker', 'confidence']]
        print(f"\n   📉 Top 5 mã GIẢM:")
        for idx, row in top_down.iterrows():
            print(f"      {row['ticker']}: {row['confidence']*100:.1f}%")
    else:
        top_down = pd.DataFrame(columns=['ticker', 'confidence'])
    
    # 8. Save predictions to CSV
    csv_path = "data/predictions.csv"
    os.makedirs("data", exist_ok=True)
    df_latest[['ticker', 'signal', 'confidence']].sort_values('confidence', ascending=False).to_csv(csv_path, index=False)
    print(f"\n   💾 Đã lưu predictions: {csv_path}")
    
    # ===== VẼ BIỂU ĐỒ (GIỮ NGUYÊN) =====
    # 9. Trực quan hóa (Vẽ biểu đồ 30 ngày gần nhất)
    if 'close' in df.columns:
        recent_prices = df['close'].values[-30:]
        label_text = 'Price Trend'
    else:
        # Fallback to first feature
        X_all = np.vstack(df['scaled_features'].apply(lambda x: x['values']).values)
        recent_prices = X_all[-30:, 0]
        label_text = 'Feature 1 (Scaled)'
    
    plt.figure(figsize=(10, 5))
    plt.plot(recent_prices, marker='o', linestyle='-', color='blue', label=label_text)
    plt.title("Biểu đồ xu hướng 30 ngày gần nhất")
    plt.xlabel("Ngày (gần nhất ở bên phải)")
    plt.ylabel("Chỉ số Giá (Đã chuẩn hóa)")
    plt.legend()
    plt.grid(True)
    
    image_path = "data/recent_trend.png"
    plt.savefig(image_path)
    plt.close()
    
    # ===== TẠO EMAIL MỚI =====
    data_info = ""
    try:
        from pyspark.sql import SparkSession
        from datetime import datetime
        spark = SparkSession.builder.appName("GetMetadata").getOrCreate()
        raw_df = spark.read.parquet("/Volumes/workspace/default/stock_data/raw/stock_data.parquet")
        
        # Get metadata
        latest_date_val = raw_df.selectExpr("max(date) as max_date").collect()[0][0]
        total_rows = raw_df.count()
        
        # Calculate days old
        latest_date_str = latest_date_val.strftime('%Y-%m-%d')
        days_old = (datetime.now() - latest_date_val).days
        
        # Determine freshness
        if days_old == 0:
            data_freshness = "🟢 MỚI NHẤT (hôm nay)"
        elif days_old == 1:
            data_freshness = "🟢 Hôm qua"
        elif days_old <= 2:
            data_freshness = f"🟡 {days_old} ngày trước"
        else:
            data_freshness = f"🔴 Cũ {days_old} ngày"
        
        data_info = f"<br><b>📅 Dữ liệu:</b> Cập nhật đến {latest_date_str} ({data_freshness}) - {total_rows:,} dòng<br>"
    except Exception as e:
        print(f"   ⚠️  Could not fetch metadata: {e}")
        data_info = "<br><b>📅 Dữ liệu:</b> Không xác định<br>"
    
    # 11. Build HTML content
    # Top UP table
    top_up_html = ""
    if len(top_up) > 0:
        top_up_html = "<h3>🔥 TOP 5 MÃ CÓ KHẢ NĂNG TĂNG CAO:</h3>"
        top_up_html += '<table border="1" style="border-collapse: collapse; width: 60%;">'
        top_up_html += '<tr style="background-color: #d4edda;"><th style="padding: 8px;">Mã</th><th style="padding: 8px;">Confidence</th></tr>'
        for idx, row in top_up.iterrows():
            top_up_html += f'<tr><td style="padding: 8px; text-align: center;"><b>{row["ticker"]}</b></td><td style="padding: 8px; text-align: center;">{row["confidence"]*100:.1f}%</td></tr>'
        top_up_html += '</table><br>'
    
    # Top DOWN table
    top_down_html = ""
    if len(top_down) > 0:
        top_down_html = "<h3>📉 TOP 5 MÃ CẦN THẬN TRỌNG (DỰ ĐOÁN GIẢM):</h3>"
        top_down_html += '<table border="1" style="border-collapse: collapse; width: 60%;">'
        top_down_html += '<tr style="background-color: #f8d7da;"><th style="padding: 8px;">Mã</th><th style="padding: 8px;">Confidence</th></tr>'
        for idx, row in top_down.iterrows():
            top_down_html += f'<tr><td style="padding: 8px; text-align: center;"><b>{row["ticker"]}</b></td><td style="padding: 8px; text-align: center;">{row["confidence"]*100:.1f}%</td></tr>'
        top_down_html += '</table><br>'
    
    # Insights
    insight_html = "<h3>💡 INSIGHTS (Top 3 yếu tố ảnh hưởng):</h3><ul>"
    for name, imp in top_features:
        insight_html += f"<li>Chỉ số <b>{name}</b> đóng góp {imp*100:.1f}% vào quyết định</li>"
    insight_html += "</ul>"
    
    # Full HTML
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #2c3e50;">📊 BÁO CÁO DỰ ĐOÁN CHỨNG KHOÁN HẰNG NGÀY</h2>
        
        <h3 style="background-color: #f0f0f0; padding: 10px;">📈 TỔNG QUAN THỊ TRƯỜNG NGÀY MAI:</h3>
        <ul style="font-size: 16px;">
            <li><b style="color: green;">{up_count}/{total} mã</b> dự đoán <b>TĂNG</b> ({up_pct:.1f}%)</li>
            <li><b style="color: red;">{down_count}/{total} mã</b> dự đoán <b>GIẢM</b> ({down_pct:.1f}%)</li>
        </ul>
        <h3 style="background-color: #fff3cd; padding: 10px;">→ Thị trường: <b>{market_sentiment}</b></h3>
        
        {top_up_html}
        
        {top_down_html}
        
        {insight_html}
        
        <h3>📊 Chi tiết đầy đủ:</h3>
        <p>Xem file đính kèm <b>predictions.csv</b> để biết dự đoán cho tất cả {total} mã cổ phiếu.</p>
        
        {data_info}
        
        <p><b>📊 Đánh giá Model:</b> Accuracy: {metrics['accuracy']:.2f}, F1-Score: {metrics['f1_score']:.2f}, AUC-ROC: {metrics['auc_roc']:.2f}</p>
        
        <p><i>Biểu đồ xu hướng đính kèm bên dưới.</i></p>
      </body>
    </html>
    """
    
    # 12. Ghi ra file HTML
    with open("data/report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   💾 Đã lưu báo cáo: data/report.html")
    
    # ===== GỬI EMAIL =====
    # 13. Gửi Email (Nếu có cấu hình trong .env)
    # Load .env file to read email config
    env_path = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops/.env"
    if not os.path.exists(env_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_env = os.path.join(project_root, ".env")
        if os.path.exists(local_env):
            env_path = local_env
        else:
            env_path = ".env"
    load_dotenv(env_path)
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    
    if sender_email and sender_password and receiver_email:
        print("\n📧 Đang gửi Email báo cáo...")
        try:
            msg = MIMEMultipart()
            
            # Subject mới
            msg['Subject'] = f"📊 Báo Cáo AI: {up_pct:.0f}% mã TĂNG ({up_count}/{total}) {sentiment_emoji}"
            msg['From'] = sender_email
            msg['To'] = receiver_email
            
            # Attach HTML
            msg.attach(MIMEText(html_content, 'html'))
            
            # Attach image
            with open(image_path, 'rb') as f:
                img_data = f.read()
                image = MIMEImage(img_data, name="trend.png")
                msg.attach(image)
            
            # Attach CSV
            with open(csv_path, 'rb') as f:
                csv_attachment = MIMEBase('application', 'octet-stream')
                csv_attachment.set_payload(f.read())
                encoders.encode_base64(csv_attachment)
                csv_attachment.add_header('Content-Disposition', f'attachment; filename="predictions.csv"')
                msg.attach(csv_attachment)
            
            # Send
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print("   ✅ Gửi Email thành công!")
        except Exception as e:
            print(f"   ❌ Lỗi gửi email: {e}")
    else:
        print("\nℹ️  Bỏ qua gửi Email vì chưa cấu hình SENDER_EMAIL trong file .env.")
