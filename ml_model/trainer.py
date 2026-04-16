import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from ml_model.model import StockGraphSAGE
from ml_model.dataset import StockDataset
import mlflow
import os

class StockTrainer:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # Đã nhận cả X, Edge_index và Edge_weight
            out = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
                pred = out.argmax(dim=1)
                
                correct += int((pred == data.y).sum())
                total += data.y.size(0)
                all_preds.append(pred)
                all_labels.append(data.y)
                
        all_labels_flat = torch.cat(all_labels).cpu().numpy()
        all_preds_flat = torch.cat(all_preds).cpu().numpy()
        
        from sklearn.metrics import f1_score
        accuracy = correct / total if total > 0 else 0
        f1 = f1_score(all_labels_flat, all_preds_flat, average='weighted', zero_division=0)
        return {"accuracy": accuracy, "f1": f1}


def run_experiment(features_path, edges_path, embeddings_path=None, centrality_path=None, epochs=10, lr=0.01, load_prev_model=True):
    """
    Hàm Khởi chạy Thực nghiệm / Warm-Start cho hệ thống MLOps.
    """
    print("--- KHỞI ĐỘNG LUỒNG HUẤN LUYỆN GNN ---")
    
    # 1. Khởi tạo Dataset
    dataset = StockDataset(features_path, edges_path, embeddings_path, centrality_path)
    if len(dataset) == 0:
        print("Lỗi: Không tìm thấy Dataset.")
        return
    
    # 2. Chia tập Train/Test theo chuỗi thời gian (80/20)
    train_size = int(0.8 * len(dataset))
    if train_size == 0: train_size = 1 # Chống lỗi nếu dataset cực nhỏ
    
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) if len(test_dataset) > 0 else train_loader
    
    # 3. Khởi tạo Khung Mô hình
    in_channels = dataset[0].x.size(1) # Tự động phát hiện kích cỡ Features
    model = StockGraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "ml_model/best_model.pt"

    # ========================================================
    # CHIẾN LƯỢC MLOPS: WARM-START (KHỞI ĐỘNG ẤM)
    # ========================================================
    if load_prev_model and os.path.exists(model_path):
        print(f"🔄 WARM-START: Tìm thấy {model_path}. Đang nạp trọng số phiên bản cũ...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Nạp thành công! Tiến hành Fine-tuning với dữ liệu lưới mới...")
            # Nếu Warm-start, số Epochs tự động giảm đi (rất nhanh)
            epochs = min(epochs, 3) 
        except Exception as e:
            print("Lỗi khi load mô hình cũ, sẽ train lại từ đầu:", e)
    else:
        print("💡 TRAINING: Không có mô hình cũ (hoặc từ chối nạp). Train toàn bộ mạng lưới từ đầu...")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = StockTrainer(model, optimizer, criterion, device=device)
    
    # 4. Ghi nhận qua MLflow
    mlflow.set_experiment("Stock_GNN_Trend_Prediction")
    
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("warm_start", load_prev_model)
        mlflow.log_param("model_type", "GraphSAGE")
        
        for epoch in range(1, epochs + 1):
            loss = trainer.train_epoch(train_loader)
            train_acc = trainer.evaluate(train_loader)
            test_acc = trainer.evaluate(test_loader)
            
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc["accuracy"], step=epoch)
            mlflow.log_metric("test_acc", test_acc["accuracy"], step=epoch)
            mlflow.log_metric("test_f1", test_acc["f1"], step=epoch)
            
            print(f"Epoch {epoch:02d}/{epochs}: Loss={loss:.4f} | Train Acc={train_acc['accuracy']:.4f} | Test Acc={test_acc['accuracy']:.4f}")
            
        # Xuất Model
        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Hoàn tất! Mô hình đã được ghi đè tại {model_path} cho pha API Inference.")
        return test_acc

if __name__ == "__main__":
    feat_p = "data/raw/stock_data - stock_data.csv"
    edge_p = "data/graph/mock_edges.csv"
    
    try:
        run_experiment(feat_p, edge_p, epochs=5, load_prev_model=True)
    except Exception as e:
        print("Lỗi khi chạy thực nghiệm (Báo cáo từ Local):", str(e))
