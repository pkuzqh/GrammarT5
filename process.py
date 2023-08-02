import sys
import os
import pickle
from Dataset import *
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':512,
    'CodeLen':512,
    'batch_size':56,
    'TableLen':100,
    'embedding_size':768,
    'WoLen':15,
    'Vocsize':100,
    'Nl_Vocsize':100,
    'max_step':3,
    'margin':0.5,
    'poolsize':50,
    'Code_Vocsize':100,
    'num_steps':50,
    'rulenum':10,
    'seed':19970316,
    'edgelen':0,
    'cnum':407,
    'mask_id':0,
    'bertnum':0,
    "gradient_accumulation_steps":10,
    "patience":5,
    "max_num_trials":10,
    "max_rel_pos":10
})

dataset = SumDataset(args, None, 'train')
dataset = SumDataset(args, None, 'test')