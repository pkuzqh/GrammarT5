from torch import nn
import torch
import torch.nn.functional as F
from gelu import GELU
from SubLayerConnection import SublayerConnection
from Multihead_Combination import   MultiHeadedCombination
class GCNN(nn.Module):
    def __init__(self, dmodel):
        super(GCNN ,self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Linear(dmodel, dmodel)
        self.linearSecond = nn.Linear(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.com = MultiHeadedCombination(8, dmodel)
    def forward1(self, state, left, inputad, edgeem=None):
        #print(state.size(), left.size())
        #if inputad.size(1) == 450:
        #state = torch.cat([left, state], dim=1)
        #rlen = inputad.size(-1)
        #state = state[rlen]
        state = self.linear(state)
        degree2 = inputad
        #print(inputad)
        #print(state.size())
        #idx = F.one_hot(degree2.long(), state.size(1))
        if left is not None:
            state = self.subconnect(state, lambda _x: self.com(_x, _x, torch.matmul(degree2, left))) #state + torch.matmul(degree2, state)
        else:
            #state = self.subconnect(state, lambda _x: self.com(_x, _x, torch.matmul(degree2.float(), state)))
            state = self.subconnect(state, lambda _x: self.com(_x, _x, torch.bmm(degree2.float(), state)))
        state = self.linearSecond(state)
        #if inputad.size(1) == 450:
        return state#self.dropout(state)[:,50:,:]
    def forward(self, state, left, inputad, edgeem=None):
        #print(state.size(), inputad.size())
        if left is not None:
            addstate = self.linear(left)
        else:
            addstate = self.linear(state)
        degree2 = inputad
        inputp = inputad.long()
        idx = torch.arange(inputp.size(0)).to(inputp.device)
        idx = idx[..., None].expand(-1, inputp.size(1))
        pem = addstate[idx, inputp, :]
        state = self.subconnect(state, lambda _x: self.com(_x, _x, pem)) #state + torch.matmul(degree2, state)
        state = self.linearSecond(state)
        return state

