import torch
import torch.nn as nn
import math
class RelEmbeddings(nn.Module):
    def __init__(self, d_model, num_heads, k, pos_type, dropout=0.0):
        super(RelEmbeddings, self).__init__()

        self.d_model = d_model
        self.k = k + 1
        self.pos_type = pos_type
        self.num_heads = num_heads
        if 'p2q' in pos_type:
            self.rel_emb_q = nn.Embedding(self.k, d_model, padding_idx=0)  # pad id=k+1 -> zero
        if 'p2k' in pos_type:
            self.rel_emb_k = nn.Embedding(self.k, d_model, padding_idx=0)
        if 'p2v' in pos_type:
            self.rel_emb_v = nn.Embedding(self.k, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def get_rel_weights(self, rel_params):
        rel_params = rel_params * math.sqrt(self.d_model)
        rel_params = self.dropout(rel_params)

        rel_params = rel_params.unsqueeze(0).unsqueeze(0)
        rel_params = rel_params.repeat(1, self.num_heads, 1, 1)

        return rel_params

    def get_p2v_emb(self, inputs):
        if 'p2v' in self.pos_type:
            rel_v = self.rel_emb_v(inputs) * math.sqrt(self.d_model)
            rel_v = self.dropout(rel_v)
            rel_v = rel_v.repeat(1, 1, 1, self.num_heads)
            return rel_v
        else:
            return None
    def forward(self):
        rel_q, rel_k, rel_v = None, None, None
        if 'p2q' in self.pos_type:
            rel_q = self.get_rel_weights(self.rel_emb_q.weight)
        if 'p2k' in self.pos_type:
            rel_k = self.get_rel_weights(self.rel_emb_k.weight)
        if 'p2v' in self.pos_type:
            rel_v = self.get_rel_weights(self.rel_emb_v.weight)

        return rel_q, rel_k, rel_v