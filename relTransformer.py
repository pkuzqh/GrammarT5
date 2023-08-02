import torch.nn as nn

from Multihead_Attention import MultiHeadedAttention
from Multihead_Attention_Rel import MultiHeadedAttentionRel
from SubLayerConnection import SublayerConnection
from DenseLayer import DenseLayer
from ConvolutionForward import ConvolutionLayer
from Multihead_Combination import MultiHeadedCombination
from TreeConv import TreeConv
from gcnn import GCNN
import torch
class rightTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention1 = MultiHeadedAttentionRel(h=attn_heads, d_model=hidden)
        self.attention2 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.Tconv_forward = GCNN(dmodel=hidden)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, inputleft, leftmask, inputParent, lefttree):
        x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))
        if x.dtype == torch.float16 and torch.isinf(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        x = self.sublayer3(x, lambda _x: self.attention2.forward(_x, inputleft, inputleft, mask=leftmask))
        if x.dtype == torch.float16 and torch.isinf(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        if lefttree:
            x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, inputleft, inputParent))
        else:
            x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputParent))
        if x.dtype == torch.float16 and torch.isinf(x).any():
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return self.dropout(x)
