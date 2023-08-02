import torch.nn as nn
from Attention import Attention
import torch
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class MultiHeadedAttentionTree(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, max_relative_position=10, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.embeddings_table = nn.Parameter(torch.Tensor(20 * 2 + 1, self.d_k))
        self.embeddings_table1 = nn.Parameter(torch.Tensor(20 * 2 + 1, self.d_k))
        nn.init.xavier_uniform_(self.embeddings_table1)
        nn.init.xavier_uniform_(self.embeddings_table)
        #self.relative_position_k = RelativePosition(self.d_k, self.max_relative_position)
        #self.relative_position_v = RelativePosition(self.d_k, self.max_relative_position)
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        qlen = query.size(1)
        klen = key.size(1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        relem = self.embeddings_table[mask.long()].cuda()
        relemv = self.embeddings_table1[mask.long()].cuda()
        rmask = torch.gt(mask, 0).unsqueeze(1)
        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=rmask, dropout=None, relem=relem, relemv=relemv)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)