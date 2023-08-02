import torch.nn as nn
from LayerNorm import LayerNorm
from run import args

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        if args.use_apex:
            from apex.normalization import FusedRMSNorm
            self.norm = FusedRMSNorm(size)
        else:
            self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer((x))))