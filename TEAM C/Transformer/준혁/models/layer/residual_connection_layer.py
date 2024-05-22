import torch.nn as nn


class ResidualConnectionLayer(nn.Module):

    # Encoder Block은 Multi-Head Attention Layer와 Position-wise Feed-Forwad Layer로 구성된다. 
    # Encoder Block을 구성하는 두 layer는 Residual Connection으로 둘러싸여 있다. 
    # output을 그대로 사용하지 않고, output에 input을 추가적으로 더한 값을 사용 (Gradient Vanishing 방지)
    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm  # Layer Normalization 
        self.dropout = nn.Dropout(p=dr_rate)  # DropOut 


    def forward(self, x, sub_layer):
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out