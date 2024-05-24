import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio):
        super().__init__()
        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x