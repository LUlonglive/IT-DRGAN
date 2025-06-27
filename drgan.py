import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from config import Config

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.residual_conv = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x, edge_index):
        residual = self.residual_conv(x)
        
        out = self.conv1(x, edge_index)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out, edge_index)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return out + residual

class DynamicGate(nn.Module):
    def __init__(self, dim):
        super(DynamicGate, self).__init__()
        self.gate = nn.Linear(dim, dim)
        
    def forward(self, current, previous):
        gate_weights = torch.sigmoid(self.gate(previous))
        return gate_weights * previous + (1 - gate_weights) * current

class DRGAN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DRGAN, self).__init__()
        self.config = Config()
        
        self.input_dim = input_dim
        self.hidden_dim = self.config.HIDDEN_DIM
        self.num_classes = num_classes
        self.num_residual_blocks = self.config.NUM_RESIDUAL_BLOCKS
        
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        
        self.residual_blocks = nn.ModuleList()
        self.dynamic_gates = nn.ModuleList()
        
        for i in range(self.num_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.config.DROPOUT_RATE)
            )
            if i > 0:
                self.dynamic_gates.append(DynamicGate(self.hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.DROPOUT_RATE),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index):
        x = self.input_projection(x)
        
        previous_features = x
        
        for i, residual_block in enumerate(self.residual_blocks):
            current_features = residual_block(previous_features, edge_index)
            
            if i > 0:
                current_features = self.dynamic_gates[i-1](current_features, previous_features)
            
            previous_features = current_features
        
        out = self.classifier(previous_features)
        return out 