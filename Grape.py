import torch.nn as nn
from gcnnnormal import GCNNM
import pickle
import numpy as np
import torch
from LayerNorm import LayerNorm
use_cuda = True
def getAd(rulead, rule):
    token2id = {}
    for x in rule:
        lst = x.split()
        if len(lst) < 2:
            continue
        #print(lst)
        token2id.setdefault(lst[0], []).append(rule[x])
    for x in rule:
        lst = x.split()
        idx1 = rule[x]
        for token in lst[2:]:
            if token not in token2id:
                continue
            for y in token2id[token]:
                idx2 = y
                rulead[idx1, idx2] = 1
                rulead[idx2, idx1] = 1
    return
def Get_Em(WordList, voc):
    ans = []
    for x in WordList:
        x = x.lower()
        if x not in voc:
            ans.append(1)
        else:
            ans.append(voc[x])
    return ans
def pad_seq(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq
def getRulePkl(rule, voc):
    inputrule = []
    for x in (rule):
        lst = x.split()
        if len(lst) < 2:
            continue 
        inputrule.append(pad_seq(Get_Em(lst, voc), 10))
    return np.array(inputrule)
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor
def filterRule(rule):
    ans = {}
    for x in rule:
        lst = x.split()
        if len(lst) < 2:
            continue
        ans[x] = len(ans)
    return ans
class Grape(nn.Module):
    def __init__(self, args, rulepath):
        super(Grape, self).__init__()
        self.rule = pickle.load(open(rulepath, "rb"))
        self.rule = filterRule(self.rule)
        #print(self.rule)
        self.rulenum = len(self.rule)
        self.voc = {'pad':0}
        self.embedding_size = args.embedding_size
        self.embedding = nn.Embedding(self.rulenum, self.embedding_size)
        self.gcnnm = GCNNM(self.embedding_size)
        self.rulead = np.zeros([self.rulenum, self.rulenum])
        getAd(self.rulead, self.rule)
        for x in self.rule:
            lst = x.split()
            if len(lst) < 2:
                continue
            for token in lst:
                if token not in self.voc:
                    self.voc[token] = len(self.voc)
        self.ruleids = getRulePkl(self.rule, self.voc)
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        self.rule_embedding = nn.Embedding(len(self.voc), self.embedding_size)
        self.norm = LayerNorm(self.embedding_size)

    def encode(self, sentence):
        return self.rule_embedding(sentence)
    def forward(self):
        self.ruleidgpu = gVar(self.ruleids).to(self.rule_embedding.weight.device)
        self.ruleidgpu = self.ruleidgpu.unsqueeze(dim=0)
        rulead = gVar(self.rulead).to(self.rule_embedding.weight.device).float()
        childEm = self.rule_embedding(self.ruleidgpu)
        childEm = self.conv(childEm.permute(0, 3, 1, 2))
        childEm = childEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        idembedding = self.embedding.weight
        x = idembedding.clone().to(self.rule_embedding.weight.device)
        for i in range(9):
            x = self.gcnnm(x, rulead, childEm).view(self.rulenum, self.embedding_size)
        return x