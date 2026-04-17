import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class StockGraphSAGE(torch.nn.Module):
    """
    Mô hình GraphSAGE cho Dự đoán Chứng khoán.
    Kiến trúc linh hoạt (Inductive Learning) phù hợp cho số lượng cổ phiếu biến động.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super(StockGraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Layer Nhập (Input)
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Các Layer Ẩn (Hidden) nếu có
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))
            self.batch_norms.append(BatchNorm(hidden_channels))
            
        # Layer Cuối (Output Convolution)
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Đầu ra Phân loại (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        """
        Luồng đi tới (Forward Pass).
        Hỗ trợ nhận tham số edge_weight (Dù GraphSAGE truyền thống dùng topology nhiều hơn).
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        logits = self.classifier(x)
        return logits
