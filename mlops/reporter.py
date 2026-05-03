import os
import numpy as np
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def generate_report(model, df, metrics):
    print("Đang tạo báo cáo Insight & Visualization...")
    
    # 1. Trích xuất Feature Importances từ XGBoost
    importances = model.feature_importances_
    
    # Gán tên giả định cho 9 features (Dựa theo chuẩn phổ biến)
    feature_names = ["MA5", "MA10", "MA20", "RSI", "MACD", "MACD_Signal", "Log_Return", "Volatility", "Price_Scaled"]
    
    # Lấy top 3 features quan trọng nhất
    top_indices = np.argsort(importances)[::-1][:3]
    top_features = [(feature_names[i], importances[i]) for i in top_indices]
    
    # 2. Dự đoán cho ngày mới nhất (Dòng cuối cùng)
    X_latest = np.array([df['scaled_features'].iloc[-1]['values']])
    prediction = model.predict(X_latest)[0]
    pred_text = "TĂNG 📈" if prediction == 1 else "GIẢM 📉"
    
    # 3. Trực quan hóa (Vẽ biểu đồ 30 ngày gần nhất)
    # Ta sẽ vẽ Feature "Price_Scaled" (cột số 8) để mô phỏng trend
    X_all = np.vstack(df['scaled_features'].apply(lambda x: x['values']).values)
    recent_prices = X_all[-30:, 8] 
    
    plt.figure(figsize=(10, 5))
    plt.plot(recent_prices, marker='o', linestyle='-', color='blue', label='Price Trend (Scaled)')
    plt.title("Biểu đồ xu hướng 30 ngày gần nhất")
    plt.xlabel("Ngày (gần nhất ở bên phải)")
    plt.ylabel("Chỉ số Giá (Đã chuẩn hóa)")
    plt.legend()
    plt.grid(True)
    
    image_path = "data/recent_trend.png"
    plt.savefig(image_path)
    plt.close()
    
    # 4. Tạo nội dung Email (HTML)
    insight_text = f"Mô hình dự đoán thị trường ngày mai sẽ <b>{pred_text}</b>.<br><br>"
    insight_text += "<b>💡 INSIGHTS (Nguyên nhân chính dẫn đến dự đoán này):</b><br>"
    for name, imp in top_features:
        insight_text += f"- Chỉ số <b>{name}</b> đóng góp {imp*100:.1f}% vào quyết định.<br>"
        
    insight_text += f"<br><b>📊 Đánh giá Model:</b> Độ chính xác (Accuracy): {metrics['accuracy']:.2f}, F1-Score: {metrics['f1_score']:.2f}"
    
    html_content = f"""
    <html>
      <body>
        <h2>BÁO CÁO DỰ ĐOÁN CHỨNG KHOÁN HẰNG NGÀY</h2>
        <p>{insight_text}</p>
        <p>Xem biểu đồ đính kèm bên dưới để biết xu hướng hiện tại.</p>
      </body>
    </html>
    """
    
    # Ghi ra file HTML để xem local nếu không gửi mail
    with open("data/report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Đã lưu báo cáo tại data/report.html và biểu đồ data/recent_trend.png")
    
    # 5. Gửi Email (Nếu có cấu hình trong .env)
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD") # App Password
    receiver_email = os.getenv("RECEIVER_EMAIL")
    
    if sender_email and sender_password and receiver_email:
        print("Đang gửi Email báo cáo...")
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"📊 Báo Cáo AI Chứng Khoán: Dự đoán {pred_text}"
            msg['From'] = sender_email
            msg['To'] = receiver_email
            
            msg.attach(MIMEText(html_content, 'html'))
            
            with open(image_path, 'rb') as f:
                img_data = f.read()
                image = MIMEImage(img_data, name="trend.png")
                msg.attach(image)
                
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print("Gửi Email thành công!")
        except Exception as e:
            print(f"Lỗi gửi email: {e}")
    else:
        print("Bỏ qua gửi Email vì chưa cấu hình SENDER_EMAIL trong file .env.")
