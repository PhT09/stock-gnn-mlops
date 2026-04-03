import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class StockGraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for Stock Trend Prediction.
    This model aggregates information from neighboring stocks (correlated stocks)
    to predict the binary trend (Up/Down) of a target stock.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super(StockGraphSAGE, self).__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
            
        # Final GraphSAGE layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.
        Args:
            x (Tensor): Node feature matrix [num_nodes, in_channels]
            edge_index (LongTensor): Graph connectivity [2, num_edges]
        Returns:
            Tensor: Prediction logits [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Apply the final classification head
        logits = self.classifier(x)
        return logits
