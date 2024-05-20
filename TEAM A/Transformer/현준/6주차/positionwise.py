import torch.nn as nn
import torch.nn.functional as F

from model.ops import init_weigth

class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(params.hidden_dim, params.feed_forward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(params.feed_forward_dim, params.hidden_dim, kernel_size=1)

        init_weigth(self.conv1)
        init_weigth(self.conv2)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        x = x.permute(0,2,1)
        output = self.dropout(self.conv1(x))
        output = self.conv2(output)
        output = output.permute(0,2,1)

        return self.dropout(output)