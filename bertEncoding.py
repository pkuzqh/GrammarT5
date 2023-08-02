import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel
class BertEm(nn.Module):
    def __init__(self):
        super(BertEm, self).__init__()
        self.model = RobertaModel.from_pretrained('roberta-base')#BertModel.from_pretrained('bert-base-uncased')
        #self.wid = torch.arange(700, 0, step=-1).float().unsqueeze(0)
    def forward(self, words):
        seqlen = words.size(1)
        batch_size = words.size(0)
        words = words.view(-1, 5 * seqlen)
        mask = torch.gt(words, 0)
        wid = torch.arange(5 * seqlen, 0, step=-1).float().unsqueeze(0).repeat(batch_size, 1).cuda()
        #print(mask.size(), wid.size())
        sortid = mask * wid
        _, idx_sort = torch.sort(sortid, dim=-1, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=1)
        words = words.gather(1, idx_sort)[:,:210]
        wordmask = torch.gt(words, 0)
        all_hidden_states = self.model(words, attention_mask=wordmask)[0]
        pad = torch.zeros(batch_size, 5 * seqlen - 210, 768).float().cuda()
        all_hidden_states = torch.cat([all_hidden_states, pad], dim=1)
        all_hidden_states = all_hidden_states.gather(1, idx_unsort.unsqueeze(-1).repeat(1, 1, 768))
        all_hidden_states = all_hidden_states.view(batch_size, seqlen, 5, 768)
        return all_hidden_states