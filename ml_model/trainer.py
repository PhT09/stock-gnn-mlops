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
            out = self.model(data.x, data.edge_index)
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
                out = self.model(data.x, data.edge_index)
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

def run_experiment(features_path, edges_path, embeddings_path, epochs=10, lr=0.01):
    # 1. Load Dataset
    dataset = StockDataset(features_path, edges_path, embeddings_path)
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return
    
    # 2. Time-Series Split (80% Train, 20% Test)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 3. Initialize Model
    # Get input dimension from the first sample
    in_channels = dataset[0].num_node_features
    model = StockGraphSAGE(in_channels=in_channels, hidden_channels=64, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = StockTrainer(model, optimizer, criterion)
    
    # 4. MLflow Logging
    mlflow.set_experiment("Stock_GNN_Trend_Prediction")
    
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("model_type", "GraphSAGE")
        
        for epoch in range(1, epochs + 1):
            loss = trainer.train_epoch(train_loader)
            train_acc = trainer.evaluate(train_loader)
            test_acc = trainer.evaluate(test_loader)
            
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc["accuracy"], step=epoch)
            mlflow.log_metric("test_acc", test_acc["accuracy"], step=epoch)
            mlflow.log_metric("test_f1", test_acc["f1"], step=epoch)
            
            print(f"Epoch {epoch:02d}: Loss={loss:.4f}, Train Acc={train_acc['accuracy']:.4f}, Test Acc={test_acc['accuracy']:.4f}")
            
        # Save Model Summary
        mlflow.pytorch.log_model(model, "model")
        torch.save(model.state_dict(), "ml_model/best_model.pt")
        print("Model saved to MLflow and local file ml_model/best_model.pt.")
        return test_acc

if __name__ == "__main__":
    # Test path
    feat = "data/processed/mock_stocks.parquet"
    edge = "data/graph/mock_edges.csv"
    emb = "data/graph/mock_embeddings.npy"
    
    run_experiment(feat, edge, emb, epochs=5)
