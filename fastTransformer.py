import torch.nn as nn

from FastAttention import FastMultiHeadedAttention
from Multihead_Attention import MultiHeadedAttention
from SubLayerConnection import SublayerConnection
from DenseLayer import DenseLayer
from ConvolutionForward import ConvolutionLayer
from Multihead_Combination import MultiHeadedCombination
import torch
class fastTransformerBlock(nn.Module):
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
        self.attention1 = FastMultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention2 = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.dense = DenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, inputleft, leftmask, start_nodes, end_nodes, rel_q, rel_k, rel_v):
        x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, start_nodes, end_nodes, rel_q, rel_k, rel_v))
        x = self.sublayer2(x, lambda _x: self.attention2.forward(_x, inputleft, inputleft, mask=leftmask))
        x = self.sublayer2(x, self.dense)
        return self.dropout(x)
