import copy
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer, norm):  # n_layer: Encoder Block의 개수 
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        self.norm = norm


    def forward(self, src, src_mask):
        out = src
        # Encoder Block들을 순서대로 실행
        # 이전 block의 output
        for layer in self.layers:
            out = layer(out, src_mask)
        out = self.norm(out)
        return out