import torch.nn as nn


class PositionWiseFeedForwardLayer(nn.Module):

    # 2개의 FC Layer를 갖는 Layer 
    # Multi-Head Attention Layer의 output을 input으로 받아 연산을 수행하고, 
    # 다음 Encoder Block에게 output을 넘겨준다.
    def __init__(self, fc1, fc2, dr_rate=0):  # (d_embed, d_ff), (d_ff, d_embed)
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = fc1   # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dr_rate)
        self.fc2 = fc2 # (d_ff, d_embed)
        # 다음 encoder block에게 shape를 유지한 채 넘겨줘야 해서 멱등 

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out