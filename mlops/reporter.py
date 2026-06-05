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
import sys
sys.path.append('/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops')

from ml_model.predict_multi_day import predict_multi_day

def generate_report(model, df, metrics, predictions_df=None):
    print("Đang tạo báo cáo Insight & Visualization...")
    
    # 1. Trích xuất Feature Importances từ XGBoost
    importances = model.feature_importances_
    
    # Gán tên cho 17 features
    feature_names = [
        "return_1d", "return_3d", "return_5d", "return_10d",
        "price_vs_ma5", "price_vs_ma10", "ma5_vs_ma10",
        "volume_ratio", "volume_change",
        "volatility_5", "volatility_10",
        "oc_return", "hl_range", "close_position",
        "return_lag1", "return_lag2", "return_lag3"
    ]
    
    # Lấy top 3 features quan trọng nhất
    top_indices = np.argsort(importances)[::-1][:3]
    top_features = [(feature_names[i], importances[i]) for i in top_indices]
    
    # ===== PHẦN MỚI: DỰ ĐOÁN 15 NGÀY =====
    print("\n🔮 Đang dự đoán 15 ngày tiếp theo...")
    
    try:
        # Use predictions_df if provided, otherwise generate new
        if predictions_df is None:
            predictions_df = predict_multi_day(n_days=15)
        
        print(f"✅ Có predictions cho {len(predictions_df)} mã × 15 ngày")
        
        # Extract day 1 data for summary
        day1_predictions = predictions_df['day_1_prediction'].values
        day1_date = predictions_df['day_1_date'].iloc[0]
        
        up_count = np.sum(day1_predictions == 1)
        down_count = np.sum(day1_predictions == 0)
        total = len(day1_predictions)
        up_pct = (up_count / total) * 100
        down_pct = 100 - up_pct
        
        # Market sentiment
        if up_pct > 60:
            market_sentiment = "XU HƯỚNG TĂNG 📈"
            sentiment_emoji = "📈"
        elif up_pct < 40:
            market_sentiment = "XU HƯỚNG GIẢM 📉"
            sentiment_emoji = "📉"
        else:
            market_sentiment = "SIDEWAY ↔️"
            sentiment_emoji = "↔️"
        
        print(f"\n   📊 Ngày 1 ({day1_date}):")
        print(f"      • {up_count}/{total} mã TĂNG ({up_pct:.1f}%)")
        print(f"      • Thị trường: {market_sentiment}")
        
        # Top stocks day 1
        top_up = predictions_df[predictions_df['day_1_prediction'] == 1].nlargest(5, 'day_1_probability')[
            ['ticker', 'day_1_probability', 'day_1_confidence']
        ]
        
        top_down = predictions_df[predictions_df['day_1_prediction'] == 0].nsmallest(5, 'day_1_probability')[
            ['ticker', 'day_1_probability', 'day_1_confidence']
        ]
        
        if len(top_up) > 0:
            print(f"\n   🔥 Top 5 mã TĂNG ngày 1:")
            for idx, row in top_up.iterrows():
                print(f"      {row['ticker']}: {row['day_1_probability']*100:.1f}% ({row['day_1_confidence']})")
        
    except Exception as e:
        print(f"\n⚠️  Lỗi khi generate predictions: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to empty predictions
        predictions_df = pd.DataFrame()
        up_count = down_count = total = 0
        up_pct = down_pct = 0
        market_sentiment = "KHÔNG XÁC ĐỊNH"
        sentiment_emoji = "❓"
        day1_date = "N/A"
        top_up = pd.DataFrame()
        top_down = pd.DataFrame()
    
    # ===== SAVE CSV 15 NGÀY =====
    csv_path = "data/predictions_15_days.csv"
    os.makedirs("data", exist_ok=True)
    
    if len(predictions_df) > 0:
        # Format CSV for easy reading
        csv_data = []
        
        for _, row in predictions_df.iterrows():
            ticker = row['ticker']
            latest_data = row['latest_data_date']
            
            # Create row with day-by-day predictions
            csv_row = {
                'Ticker': ticker,
                'Latest_Data': latest_data
            }
            
            for day in range(1, 16):
                date_col = f'day_{day}_date'
                signal_col = f'day_{day}_signal'
                prob_col = f'day_{day}_probability'
                conf_col = f'day_{day}_confidence'
                
                csv_row[f'Day{day}_Date'] = row[date_col]
                csv_row[f'Day{day}_Signal'] = row[signal_col]
                csv_row[f'Day{day}_Confidence'] = f"{row[prob_col]*100:.1f}%"
                csv_row[f'Day{day}_Level'] = row[conf_col]
            
            csv_data.append(csv_row)
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n   💾 Đã lưu CSV: {csv_path}")
        print(f"      • {len(csv_df)} tickers")
        print(f"      • 15 ngày predictions cho mỗi mã")
    
    # ===== VẼ BIỂU ĐỒ =====
    # NEW: Use individual scaled columns instead of 'scaled_features' vector
    # Use volume_ratio_scaled as a proxy for recent trend visualization
    try:
        if 'volume_ratio_scaled' in df.columns:
            recent_prices = df['volume_ratio_scaled'].values[-30:]
            label_text = 'Volume Ratio Trend (Scaled)'
        elif 'close' in df.columns:
            recent_prices = df['close'].values[-30:]
            label_text = 'Price Trend'
        else:
            # Fallback: use first scaled feature
            recent_prices = df['return_1d_scaled'].values[-30:]
            label_text = 'Return 1D (Scaled)'
    except Exception as e:
        print(f"\n⚠️  Could not extract chart data: {e}")
        recent_prices = np.zeros(30)
        label_text = 'No Data'
    
    plt.figure(figsize=(10, 5))
    plt.plot(recent_prices, marker='o', linestyle='-', color='blue', label=label_text)
    plt.title("Biểu đồ xu hướng 30 ngày gần nhất")
    plt.xlabel("Ngày (gần nhất ở bên phải)")
    plt.ylabel("Chỉ số (Đã chuẩn hóa)")
    plt.legend()
    plt.grid(True)
    
    image_path = "data/recent_trend.png"
    plt.savefig(image_path)
    plt.close()
    
    # ===== METADATA =====
    data_info = ""
    try:
        from pyspark.sql import SparkSession
        from datetime import datetime
        spark = SparkSession.builder.appName("GetMetadata").getOrCreate()
        raw_df = spark.read.parquet("/Volumes/workspace/default/stock_data/raw/stock_data.parquet")
        
        latest_date_val = raw_df.selectExpr("max(date) as max_date").collect()[0][0]
        total_rows = raw_df.count()
        
        latest_date_str = latest_date_val.strftime('%Y-%m-%d')
        days_old = (datetime.now() - latest_date_val).days
        
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
    
    # ===== BUILD EMAIL HTML =====
    # Top UP table (Day 1)
    top_up_html = ""
    if len(top_up) > 0:
        top_up_html = "<h3>🔥 TOP 5 MÃ CÓ KHẢ NĂNG TĂNG CAO (NGÀY 1):</h3>"
        top_up_html += '<table border="1" style="border-collapse: collapse; width: 70%;">'
        top_up_html += '<tr style="background-color: #d4edda;"><th style="padding: 8px;">Mã</th><th style="padding: 8px;">Confidence</th><th style="padding: 8px;">Level</th></tr>'
        for idx, row in top_up.iterrows():
            top_up_html += f'<tr><td style="padding: 8px; text-align: center;"><b>{row["ticker"]}</b></td>'
            top_up_html += f'<td style="padding: 8px; text-align: center;">{row["day_1_probability"]*100:.1f}%</td>'
            top_up_html += f'<td style="padding: 8px; text-align: center;">{row["day_1_confidence"]}</td></tr>'
        top_up_html += '</table><br>'
    
    # Top DOWN table (Day 1)
    top_down_html = ""
    if len(top_down) > 0:
        top_down_html = "<h3>📉 TOP 5 MÃ CẦN THẬN TRỌNG (NGÀY 1):</h3>"
        top_down_html += '<table border="1" style="border-collapse: collapse; width: 70%;">'
        top_down_html += '<tr style="background-color: #f8d7da;"><th style="padding: 8px;">Mã</th><th style="padding: 8px;">Confidence</th><th style="padding: 8px;">Level</th></tr>'
        for idx, row in top_down.iterrows():
            top_down_html += f'<tr><td style="padding: 8px; text-align: center;"><b>{row["ticker"]}</b></td>'
            top_down_html += f'<td style="padding: 8px; text-align: center;">{row["day_1_probability"]*100:.1f}%</td>'
            top_down_html += f'<td style="padding: 8px; text-align: center;">{row["day_1_confidence"]}</td></tr>'
        top_down_html += '</table><br>'
    
    # Week overview (Days 1-5)
    week_html = ""
    if len(predictions_df) > 0:
        week_html = "<h3>📅 DỰ ĐOÁN TUẦN NÀY (5 NGÀY ĐẦU):</h3>"
        week_html += '<table border="1" style="border-collapse: collapse; width: 80%;">'
        week_html += '<tr style="background-color: #e7f3ff;"><th style="padding: 8px;">Ngày</th><th style="padding: 8px;">📈 TĂNG</th><th style="padding: 8px;">📉 GIẢM</th><th style="padding: 8px;">Xu hướng</th></tr>'
        
        for day in range(1, 6):
            pred_col = f'day_{day}_prediction'
            date_col = f'day_{day}_date'
            
            date_val = predictions_df[date_col].iloc[0]
            up = np.sum(predictions_df[pred_col] == 1)
            down = np.sum(predictions_df[pred_col] == 0)
            up_ratio = up / total * 100
            
            if up_ratio > 55:
                trend = "📈 Tăng"
                bg = "#d4edda"
            elif up_ratio < 45:
                trend = "📉 Giảm"
                bg = "#f8d7da"
            else:
                trend = "↔️ Sideway"
                bg = "#fff3cd"
            
            week_html += f'<tr style="background-color: {bg};"><td style="padding: 8px;"><b>{date_val}</b></td>'
            week_html += f'<td style="padding: 8px; text-align: center;">{up} ({up_ratio:.0f}%)</td>'
            week_html += f'<td style="padding: 8px; text-align: center;">{down} ({100-up_ratio:.0f}%)</td>'
            week_html += f'<td style="padding: 8px; text-align: center;"><b>{trend}</b></td></tr>'
        
        week_html += '</table><br>'
    
    # Insights
    insight_html = "<h3>💡 INSIGHTS (Top 3 yếu tố ảnh hưởng):</h3><ul>"
    for name, imp in top_features:
        insight_html += f"<li>Chỉ số <b>{name}</b> đóng góp {imp*100:.1f}% vào quyết định</li>"
    insight_html += "</ul>"
    
    # Full HTML
    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #2c3e50;">📊 BÁO CÁO DỰ ĐOÁN CHỨNG KHOÁN 15 NGÀY</h2>
        
        <h3 style="background-color: #f0f0f0; padding: 10px;">📈 TỔNG QUAN NGÀY MAI ({day1_date}):</h3>
        <ul style="font-size: 16px;">
            <li><b style="color: green;">{up_count}/{total} mã</b> dự đoán <b>TĂNG</b> ({up_pct:.1f}%)</li>
            <li><b style="color: red;">{down_count}/{total} mã</b> dự đoán <b>GIẢM</b> ({down_pct:.1f}%)</li>
        </ul>
        <h3 style="background-color: #fff3cd; padding: 10px;">→ Thị trường: <b>{market_sentiment}</b></h3>
        
        {top_up_html}
        
        {top_down_html}
        
        {week_html}
        
        {insight_html}
        
        <h3>📊 Chi tiết đầy đủ:</h3>
        <p><b>🎁 File đính kèm <span style="color: red;">predictions_15_days.csv</span> chứa:</b></p>
        <ul>
            <li>✅ <b>TẤT CẢ {total} mã cổ phiếu</b></li>
            <li>✅ <b>Dự đoán 15 NGÀY giao dịch tiếp theo</b></li>
            <li>✅ Signal (TĂNG/GIẢM), Confidence (%), Level (HIGH/MEDIUM/LOW)</li>
        </ul>
        <p style="background-color: #e7f3ff; padding: 10px; border-left: 4px solid #0066cc;">
            <b>💡 Cách dùng:</b> Mở file CSV bằng Excel/Google Sheets để xem dự đoán chi tiết cho từng mã trong 15 ngày tới.
        </p>
        
        {data_info}
        
        <p><b>📊 Đánh giá Model:</b> Accuracy: {metrics['accuracy']:.2f}, F1-Score: {metrics['f1_score']:.2f}, AUC-ROC: {metrics['auc_roc']:.2f}</p>
        
        <h3>📈 Biểu đồ xu hướng 30 ngày gần nhất:</h3>
        <p><i>Xem file ảnh đính kèm</i></p>
        
        <hr>
        <p style="color: gray; font-size: 12px;">
        Báo cáo tự động từ MLOps Pipeline | Databricks | Confidential<br>
        <b>⚠️ Lưu ý:</b> Đây là dự đoán từ model Machine Learning, không phải lời khuyên đầu tư. Vui lòng tham khảo thêm các nguồn khác.
        </p>
      </body>
    </html>
    """
    
    return html_content, image_path, csv_path


def send_email(subject, html_content, image_path, csv_path):
    """Send email with HTML report, chart image, and CSV attachments"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Email config từ .env
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    receiver_emails_str = os.getenv("RECEIVER_EMAILS", "")
    
    if not sender_email or not sender_password:
        print("⚠️  Email credentials not found in .env - skipping email send")
        return False
    
    receiver_emails = [email.strip() for email in receiver_emails_str.split(",") if email.strip()]
    
    if not receiver_emails:
        print("⚠️  No receiver emails configured - skipping email send")
        return False
    
    print(f"\n📧 Sending email to: {', '.join(receiver_emails)}")
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_emails)
    msg['Subject'] = subject
    
    # Attach HTML body
    msg.attach(MIMEText(html_content, 'html'))
    
    # Attach chart image
    if os.path.exists(image_path):
        with open(image_path, 'rb') as img:
            mime_img = MIMEImage(img.read())
            mime_img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
            msg.attach(mime_img)
        print(f"   ✅ Attached image: {image_path}")
    
    # Attach CSV file
    if os.path.exists(csv_path):
        with open(csv_path, 'rb') as csv_file:
            mime_csv = MIMEBase('application', 'octet-stream')
            mime_csv.set_payload(csv_file.read())
            encoders.encode_base64(mime_csv)
            mime_csv.add_header('Content-Disposition', 'attachment', filename=os.path.basename(csv_path))
            msg.attach(mime_csv)
        print(f"   ✅ Attached CSV: {csv_path}")
    
    # Send email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_emails, msg.as_string())
        print(f"✅ Email sent successfully to {len(receiver_emails)} recipient(s)")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        import traceback
        traceback.print_exc()
        return False
