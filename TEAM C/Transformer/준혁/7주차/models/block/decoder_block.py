import copy
import torch.nn as nn

from models.layer.residual_connection_layer import ResidualConnectionLayer


class DecoderBlock(nn.Module):

    # Encoder에서 넘어오는 context가 각 Decoder Block마다 input으로 주어진다
    # Encoder Block과 달리 Multi-Head Attention Layer가 2개가 존재
    # (1) Self-Multi-Head Attention Layer: Decoder의 input으로 주어지는 sentence 내부에서의 Attention을 계산
    # (2) Cross-Multi-Head Attention Layer
    # Decoder Block 내부에서 전달된 input(Self-Multi-Head Attention Layer의 output)은 Query로 사용하고, 
    # Encoder에서 넘어온 context는 Key와 Value로 사용
    # (3) Position-wise Feed-Forward Layer
    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate=0):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.cross_attention = cross_attention
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.position_ff = position_ff
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)


    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residual3(out, self.position_ff)
        return out