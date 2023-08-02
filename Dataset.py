import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
import json
from copy import deepcopy
from scipy import sparse
from transformers import AutoTokenizer
import math
from tqdm import tqdm
import torch
from torch.utils.data import Sampler
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
tokenlen = len(tokenizer.get_vocab())
args = {}
voc = {}
idenlst = [32104, 32115, 32127, 32132, 32138, 32154, 32158, 32171, 32176, 32179, 32185, 32193, 32197, 32198, 32200, 32214, 32227, 32231, 32247, 32255, 32261, 32266, 32270, 32274, 32285, 32290, 32305, 32312, 32313, 32322, 32334, 32344, 32347, 32349, 32350, 32352, 32359, 32362, 32374, 32405, 32413, 32414, 32436, 32459, 32466, 32472, 32474, 32475, 32484, 32500, 32525, 32528, 32530, 32542, 32581, 32586, 32593, 32594, 32599, 32603, 32608, 32610, 32612, 32621, 32636, 32639, 32643, 32644, 32656, 32657, 32671, 32679, 32682, 32685, 32687, 32707, 32730, 32761, 32762, 32770, 32790, 32801, 32802, 32804, 32807, 32809, 32810, 32813, 32814, 32820, 32835, 32845, 32849, 32853, 32857, 32860, 32862, 32865, 32870, 32883, 32884, 32888, 32898, 32902, 32930, 32932, 32938, 32944, 32947, 32948, 32950, 32951, 32953, 32954, 32955, 32957, 32960, 32972, 32974, 32979, 32980, 32983, 32989, 32994, 33002, 33005, 33013, 33018, 33020, 33038, 33043, 33046, 33048, 33058, 33059, 33066, 33069, 33084, 33096, 33097, 33100, 33119, 33126, 33144, 33151, 33163, 33165, 33191, 33198, 33202, 33205, 33211, 33215, 33227, 33246, 33290, 33308, 33310, 33316, 33345, 33350, 33352, 33356, 33367, 33384, 33390, 33392, 33413, 33417, 33419, 33429, 33441, 33473, 33510, 33539, 33540, 33545, 33589, 33639, 33644, 33653, 33705, 33747, 33757, 33760, 33787, 33788, 33789, 33811, 33813, 33814, 33815, 33820, 33821, 33851, 33868, 33889, 33900, 33911, 33912, 33915, 33916, 33917, 33918, 33921, 33946, 33947, 33959, 33976, 33988, 33989, 34000, 34005, 34010, 34024, 34032, 34040, 34062, 34065, 34076, 34088, 34097, 34109, 34118, 34119, 34120, 34125, 34127, 34135, 34161, 34175, 34199, 34202, 34213, 34229, 34247, 34267, 34289, 34304, 34333, 34342, 34352, 34374, 34387, 34391, 34397, 34408, 34410, 34414, 34416, 34421, 34422, 34427, 34428, 34429, 34452, 34486, 34492, 34505, 34535, 34547, 34565, 34591, 34611, 34626, 34632, 34637, 34645, 34649, 34658, 34660, 34692, 34715, 34738, 34753, 34756, 34759, 34768, 34783, 34791, 34815, 34827, 34835, 34861, 34863, 34864, 34894, 34928, 34931, 34959, 34987, 35024, 35055, 35086, 35132, 35143, 35151, 35184, 35208, 35240, 35260, 35266, 35271, 35287, 35291, 35310, 35316, 35347, 35356, 35372, 35385, 35390, 35423, 35425, 35429, 35430, 35432, 35447, 35448, 35461, 35462, 35463, 35466, 35492, 35517, 35526, 35533, 35547, 35551, 35565, 35566, 35574, 35587, 35590, 35594, 35616, 35631, 35655, 35660, 35670, 35671, 35675, 35691, 35702, 35707, 35733, 35757, 35763, 35785, 35810, 35814, 35820, 35824, 35829, 35852, 35854, 35866, 35879, 35882, 35886, 35894, 35901, 35925, 35946, 35951, 35952, 35977, 35978, 35988, 35989, 35990, 35994, 36001, 36008, 36032, 36059, 36062, 36071]

class Graph:
    def __init__(self):
        self.row = []
        self.col = []
        self.val = []
        self.edge = {}
        self.rowNum = 0
        self.colNum = 0
    def addEdge(self, r, c, v):
        if (r, c) in self.edge:
            print(r, c)
            assert(0)
        self.edge[(r, c)] = len(self.row)
        self.row.append(r)
        self.col.append(c)
        self.val.append(v)
        '''self.edge[(c, r)] = len(self.row)
        self.row.append(c)
        self.col.append(r)
        self.val.append(v)'''
    def editVal(self, r, c, v):
        self.val[self.edge(r, c)] = v
    def updateval(self, index, v):
        self.val[index] = v
    def normlize(self):
        r = {}
        c = {}
        for i  in range(len(self.row)):
            if self.row[i] not in r:
                r[self.row[i]] = 0
            r[self.row[i]] += 1
            #if self.col[i] not in c:
            #    c[self.col[i]] = 0
            #c[self.col[i]] += 1
        for i in range(len(self.row)):
            if self.row[i] in r:
                self.val[i] /= r[self.row[i]]
            #self.val[i] = 1 / r[self.row[i]]
    def tonumpy(self, r, c):
        ans = np.zeros([r, c])
        for i in range(len(self.row)):
            ans[self.row[i], self.col[i]] = self.val[i]
        return ans
    def tonumpyRel(self, r, relpos):
        ans = -np.ones([relpos, r])
        for i in range(len(self.row)):
            if self.row[i] < r and self.col[i] < r:
                ans[self.val[i], self.row[i]] = self.col[i]
        return ans
def splitCamel(s):
    if s.isupper():
        return [s.lower()]
    ans = []
    tmpans = ""
    for x in s:
        if x.isupper() or x == '_':
            if tmpans != "":
                ans.append(tmpans)
            tmpans = x.replace("_", "")
        else:
            tmpans += x
    if tmpans != "":
        ans.append(tmpans)
    for i in range(len(ans)):
        ans[i] = ans[i].lower()
    return ans
def isnum(str_number):
    return (str_number.split(".")[0]).isdigit() or str_number.isdigit() or  (str_number.split('-')[-1]).split(".")[-1].isdigit()
PAD_token = tokenizer.pad_token_id
def pad_seq(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [PAD_token] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq
def pad_seq2(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq

def pad_list(seq, maxlen1, maxlen2):
    if len(seq) < maxlen1:
        seq = seq + [[PAD_token] * maxlen2] * maxlen1
        seq = seq[:maxlen1]
    else:
        seq = seq[:maxlen1]
    return seq
def mask_span(inputnl, inputparent, ratio=0.15):
    global voc
    def verify(mask_span_start_id,span_length,masked_positions,seq_len):
        flag = True
        if mask_span_start_id-1 in masked_positions:
            return False
        for i in range(span_length+1):
            if mask_span_start_id+i in masked_positions or mask_span_start_id+i >= seq_len:
                flag = False
        return flag
    seq_len = len(inputnl)
    num_spans = max(1, random.randint(round(seq_len*ratio)-1,round(seq_len*ratio)+1))
    num_spans = min(num_spans, 50)
    masked_positions = []
    masked_start_positions = []
    masked_span_lengths = []
    total_span_length = 0
    for _ in range(num_spans):
        mask_span_start_id = random.randint(0, seq_len-1)
        span_length = random.randint(1, 5)
        trial = 0
        while not verify(mask_span_start_id,span_length,masked_positions,seq_len):
            mask_span_start_id = random.randint(0, seq_len-1)
            span_length = random.randint(1, 5)
            trial += 1
            if trial > 15:
                break
        if trial > 15:
            break
        for i in range(span_length):
            masked_positions.append(mask_span_start_id+i)
            masked_start_positions.append(mask_span_start_id)
            masked_span_lengths.append(span_length)
            total_span_length += span_length
    new_inputnl = []
    newinputparent = []
    newnlparent = []
    inputres = []
    spanid = 0
    voc = tokenizer.get_vocab()
    for i in range(seq_len):
        if i in masked_start_positions:
            new_inputnl.append(voc["<extra_id_%d>"%(spanid)])
            inputres.append(voc["<extra_id_%d>"%(spanid)])
            spanid += 1
            inputres.append(inputnl[i])

        elif i in masked_positions:
            inputres.append(inputnl[i])
        else:
            new_inputnl.append(inputnl[i])
            
    return new_inputnl, inputres
def mask_subtree(inputnl, inputendlist, ratio=0.02):
    global voc
    def verify(mask_span_start_id,span_length,masked_positions,seq_len):
        flag = True
        if mask_span_start_id-1 in masked_positions:
            return False
        for i in range(span_length+1):
            if mask_span_start_id+i in masked_positions or mask_span_start_id+i >= seq_len:
                flag = False
        return flag
    validposition = []
    secondposition = []
    secondlens = []
    seqlens = []
    for i in range(len(inputendlist)):
        length = inputendlist[i] - i + 1
        if length >= 10 and length <= 60:
            validposition.append(i)
            seqlens.append(length)
        if length >= 5:
            secondposition.append(i)
            secondlens.append(length)
    if len(validposition) == 0:
        validposition = secondposition
        seqlens = secondlens
    seq_len = len(inputnl)
    num_spans = max(1, random.randint(round(seq_len*ratio)-1,round(seq_len*ratio)+1))
    num_spans = min(num_spans, 50)
    masked_positions = []
    masked_start_positions = []
    for _ in range(num_spans):
        idx = random.randint(0, len(validposition)-1)
        mask_span_start_id = validposition[idx]
        span_length = seqlens[idx]
        trial = 0
        while not verify(mask_span_start_id,span_length,masked_positions,seq_len):
            idx = random.randint(0, len(validposition)-1)
            mask_span_start_id = validposition[idx]
            span_length = seqlens[idx]
            trial += 1
            if trial > 15:
                break
        if trial > 15:
            break
        for i in range(span_length):
            masked_positions.append(mask_span_start_id+i)
            masked_start_positions.append(mask_span_start_id)
    new_inputnl = []
    inputres = []
    spanid = 0
    voc = tokenizer.get_vocab()
    for i in range(seq_len):
        if i in masked_start_positions:
            new_inputnl.append(voc["<extra_id_%d>"%(spanid)])
            inputres.append(voc["<extra_id_%d>"%(spanid)])
            spanid += 1
            inputres.append(inputnl[i])
        elif i in masked_positions:
            inputres.append(inputnl[i])
        else:
            new_inputnl.append(inputnl[i])
            
    return new_inputnl, inputres
def mask_iden(inputnl, inputparent):
    global tokenlen, voc
    rrdict = {}
    for x in voc:
        rrdict[voc[x]] = x
    number = 0
    startidx = -1
    newinputnl = []
    newinputparent = []
    newnlparent = []
    inputres = []
    coresid = {}
    for i in (range(1, len(inputparent))):
        if number >= 100:
            break
        if inputnl[i] < tokenlen and inputnl[inputparent[i]] in idenlst:
            if startidx == -1:
                startidx = i
                newinputnl.append(voc["<extra_id_%d>"%(number)])
                inputres.append(voc["<extra_id_%d>"%(number)])
                number += 1
                inputres.append(inputnl[i])
            else:
                inputres.append(inputnl[i])
        else:
            if startidx != -1:
                startidx = -1
            newinputnl.append(inputnl[i])
    return newinputnl, inputres
class ChunkedRandomSampler(Sampler):

    def __init__(self, data_source, batch_size):
      self.data_source = data_source
      self.batch_size = batch_size

    def __iter__(self):
      lst = list(range(len(self.data_source)))
      chunked = [lst[i:i+self.batch_size] for i in range(0, len(self.data_source), self.batch_size)]
      random.shuffle(chunked)
      new_lst = [e for piece in chunked for e in piece]
      return iter(new_lst)

    def __len__(self):
      return len(self.data_source)
def rs_collate_fn(batch):
    tbatch = {}
    rbatch = {}
    maxnllen = 0
    maxcodelen = 0
    bnl = []
    bnlparent = []
    bparent = []
    bres = []

    inl = []
    inlparent = []
    iparent = []
    ires = []

    snl = []
    sres = []

    maxnllen1 = 0
    maxcodelen1 = 0

    maxnllen2 = 0
    maxcodelen2 = 0

    maxnllen3 = 0
    maxcodelen3 = 0
    masktype = random.randint(0, 2)
    for k in (range(len(batch))):
        if 'nl' in batch[k]:
            inputnlcode = batch[k]['rulelist'][:-1][:513]
            fatherlist = batch[k]['fatherlist'][:-1][:513]
            inputnl = batch[k]['nl'][:-1][:512]
            endlist = batch[k]['endlist'][:-1][:513]
        else:
            inputnlcode = batch[k]['rulelist'][:-1][:513]
            fatherlist = batch[k]['fatherlist'][:-1][:513]
            endlist = batch[k]['endlist'][:-1][:513]    
            inputnl = []
        #autoencode
        if masktype == 3:
            inputnlo = [tokenizer.cls_token_id] + inputnl + [tokenizer.sep_token_id] + inputnlcode[1:] + [tokenizer.sep_token_id]
            inputres = inputnlo
            bnl.append(inputnlo)
            bres.append(inputres)
            maxnllen3 = max(maxnllen3, len(inputnlo))
            maxcodelen3 = max(maxcodelen3, len(inputres))
        #mask span noise
        if masktype == 0:
            inputnlo = [tokenizer.cls_token_id] + inputnl + [tokenizer.sep_token_id] + inputnlcode[1:] + [tokenizer.sep_token_id]
            inputnlo, inputres = mask_span(inputnlo, fatherlist, ratio=0.15)
            bnl.append(inputnlo)
            bres.append(inputres)        
            maxnllen1 = max(maxnllen1, len(inputnlo))
            maxcodelen1 = max(maxcodelen1, len(inputres))
        #mask iden noise
        if masktype == 1:
            inputnlcode, inputres = mask_iden(inputnlcode, fatherlist)
            inputnlo = [tokenizer.cls_token_id] + inputnl + [tokenizer.sep_token_id] + inputnlcode[1:] + [tokenizer.sep_token_id]
            inl.append(inputnlo)
            ires.append(inputres)
            maxnllen2 = max(maxnllen2, len(inputnlo))
            maxcodelen2 = max(maxcodelen2, len(inputres))
        #mask subtree noise
        if masktype == 2:
            inputnlcode, inputres = mask_subtree(inputnlcode, endlist)
            inputnlo = [tokenizer.cls_token_id] + inputnl + [tokenizer.sep_token_id] + inputnlcode[1:] + [tokenizer.sep_token_id]
            snl.append(inputnlo)
            sres.append(inputres)
            maxnllen3 = max(maxnllen3, len(inputnlo))
            maxcodelen3 = max(maxcodelen3, len(inputres))

    maxnllen1 = min(maxnllen1, 512)
    maxnllen2 = min(maxnllen2, 512)
    maxcodelen1 = min(maxcodelen1, 256)
    maxcodelen2 = min(maxcodelen2, 256)
    maxnllen3 = min(maxnllen3, 512)
    maxcodelen3 = min(maxcodelen3, 256)
    lens = 0
    if masktype == 0:
        lens = len(bnl)
    if masktype == 1:
        lens = len(inl)
    if masktype == 2:
        lens = len(snl)
    if masktype == 3:
        lens = len(bnl)
    for i in range(lens):
        if masktype == 0:
            bnl[i] = pad_seq(bnl[i], maxnllen1)
            bres[i] = pad_seq(bres[i], maxcodelen1)
        if masktype == 1:
            inl[i] = pad_seq(inl[i], maxnllen2)
            ires[i] = pad_seq(ires[i], maxcodelen2)
        if masktype == 2:
            snl[i] = pad_seq(snl[i], maxnllen3)
            sres[i] = pad_seq(sres[i], maxcodelen3)
        if masktype == 3:
            bnl[i] = pad_seq(bnl[i], maxnllen3)
            bres[i] = pad_seq(bres[i], maxcodelen3)
    if masktype == 0:
        tbatch['nl'] = torch.tensor(bnl)
        tbatch['res'] = torch.tensor(bres)
    if masktype == 1:
        tbatch['nl'] = torch.tensor(inl)
        tbatch['res'] = torch.tensor(ires)
    if masktype == 2:
        tbatch['nl'] = torch.tensor(snl)
        tbatch['res'] = torch.tensor(sres)
    if masktype == 3:
        tbatch['nl'] = torch.tensor(bnl)
        tbatch['res'] = torch.tensor(bres)
    return tbatch
def rs_collate_fn1(batch):
    rbatch = {}
    bnl = []
    bparent = []
    bres = []
    bnlparent = []

    maxnllen1 = 0
    maxcodelen1 = 0


    for k in (range(len(batch))):
        inputnl = batch[k]['nl']
        #batch[k]['nl'][:-1]
        #nlparent = [i for i in range(len(inputnl))]
        inputres = batch[k]['rulelist'][1:-1]
        #inputparent = batch[k]['fatherlist']
        
        bnl.append(inputnl)
        #bnlparent.append(nlparent)
        #bparent.append(inputparent)
        bres.append(inputres)        
        maxnllen1 = max(maxnllen1, len(inputnl))
        maxcodelen1 = max(maxcodelen1, len(inputres))
    maxnllen1 = min(maxnllen1, args.NlLen)
    maxcodelen1 = min(maxcodelen1, args.CodeLen)
    for i in range(len(bnl)):
        bnl[i] = pad_seq(bnl[i], maxnllen1)
        #bnlparent[i] = pad_seq2(bnlparent[i], maxnllen1)
        #bparent[i] = pad_seq2(bparent[i], maxcodelen1)
        bres[i] = pad_seq(bres[i], maxcodelen1)
    rbatch['nl'] = torch.tensor(bnl)
    #rbatch['nlparent'] = torch.tensor(bnlparent)
    #rbatch['parent'] = torch.tensor(bparent)
    rbatch['res'] = torch.tensor(bres)
    return rbatch
def pad_seq_with(seq, maxlen):
    seq = [x if x != 2 else 1 for x in seq]
    if len(seq) < maxlen:
        seq = [1] + seq + [2] + [PAD_token] * maxlen
        seq = seq[:maxlen + 2]
    else:
        seq = seq[:maxlen]
        seq = [1] + seq + [2]
    return seq
def rs_collate_fn2(batch):
    rbatch = {}
    bnl = []
    bparent = []
    bres = []
    bnlparent = []

    maxnllen1 = 0
    maxcodelen1 = 0


    for k in (range(len(batch))):
        inputnl = batch[k]['nl'][1:-1]
        #batch[k]['nl'][:-1]
        #nlparent = [i for i in range(len(inputnl))]
        inputres = batch[k]['rulelist'][2:-2]
        #inputparent = batch[k]['fatherlist']
        
        bnl.append(inputnl)
        #bnlparent.append(nlparent)
        #bparent.append(inputparent)
        bres.append(inputres)        
        maxnllen1 = max(maxnllen1, len(inputnl))
        maxcodelen1 = max(maxcodelen1, len(inputres))
    maxnllen1 = min(maxnllen1, args.NlLen)
    maxcodelen1 = min(maxcodelen1, args.CodeLen)
    for i in range(len(bnl)):
        bnl[i] = pad_seq_with(bnl[i], maxnllen1)
        #bnlparent[i] = pad_seq2(bnlparent[i], maxnllen1)
        #bparent[i] = pad_seq2(bparent[i], maxcodelen1)
        bres[i] = pad_seq_with(bres[i], maxcodelen1)
    rbatch['nl'] = torch.tensor(bnl)
    #rbatch['nlparent'] = torch.tensor(bnlparent)
    #rbatch['parent'] = torch.tensor(bparent)
    rbatch['res'] = torch.tensor(bres)
    return rbatch
def readpickle(filename, debug=False):
    data = []
    pbar = tqdm()
    with open(filename, 'rb') as f:
        while True:
            if len(data) % 100000 == 0:
                print("%d data read" % len(data))
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
            if not isinstance(data[-1], dict):
                data = data[:-1]
            if debug and len(data) > 1000:
                break
            pbar.update(1)
    pbar.close()
    return data
class SumDataset(data.Dataset):
    def __init__(self, config, code_voc=None, dataName="train", idx=-1, mode='mask'):
        global args
        global voc
        args = config
        self.train_path = "processdata.pkl"
        self.val_path = "test1_process.txt"  # "validD.txt"
        self.test_path = "dev_process.txt"
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = code_voc
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Table_Len = config.TableLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        config.mask_id = tokenizer.pad_token_id
        config.bertnum = len(tokenizer.get_vocab())
        self.bertnum = config.bertnum
        self.max_rel_pos = args.max_rel_pos
        self.PAD_token = config.mask_id
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        self.num_step = 50
        self.ruledict = pickle.load(open("csharprule.pkl", "rb"))
        #self.ruledict["start -> root"] = len(self.ruledict)
        self.rrdict = {}
        self.tablename = {}
        self.cache = {}
        for x in self.ruledict:
            self.rrdict[self.ruledict[x]] = x
        tokens = []
        for i in range(self.bertnum, len(self.rrdict)):
            tokens.append(self.rrdict[i])
        tokenizer.add_tokens(tokens)
        #tokenizer.add_tokens([self.rrdict[i]])
        #tmp = []
        #for i in range(50):
        #    tmp.append('[MASK%d]' % i)
        #tokenizer.add_tokens(tmp)
        voc = tokenizer.get_vocab()
        self.rulenum = len(voc)
        if dataName == "train":
            if mode == 'mask':
                assert os.path.exists('traindata%d.pkl'%idx)
                self.data = readpickle('traindata%d.pkl'%idx, debug=False)
                #pickle.load(open('traindata%d.pkl'%idx, "rb"))
                self.data.sort(key=lambda x: len(x['rulelist']), reverse=True)
                #self.data = self.preProcessData(pickle.load(open('rawtraindata.pkl', 'rb')))
            elif mode == 'finetune':
                self.data = pickle.load(open('fttrain%d.pkl'%idx, "rb"))
                self.data.sort(key=lambda x: len(x['rulelist']), reverse=True)
                leng = []
                leng2 = []
                for x in self.data:
                    leng.append(len(x['rulelist']))
                    leng2.append(len(x['nl']))
                import numpy as np
                print(np.mean(leng), np.mean(leng2))
            elif mode == 'finetunesearch':
                self.data = pickle.load(open('fttrain%d.pkl'%idx, "rb"))
            else:
                self.data = readpickle('traindatawithnl%d.pkl'%idx, debug=False)
                self.data.sort(key=lambda x: len(x['rulelist']), reverse=True)
        elif dataName == "valid":
            self.data = pickle.load(open('ftvalid%d.pkl'%idx, 'rb'))
        elif dataName == "eval":
            if os.path.exists("evaldata.pkl"):
                self.data = pickle.load(open("evaldata.pkl", "rb"))
                self.nl = pickle.load(open("evalnl.pkl", "rb"))
                self.tabless = pickle.load(open("evaltable.pkl", "rb"))
                return
            self.data = self.preProcessDataEval(open('eval.txt', "r", encoding='utf-8'))
        elif dataName == "testone":
            pass
        elif dataName == 'testbase':
            self.data = pickle.load(open('fttestbase%d.pkl'%idx, 'rb'))
        else:
            if mode == 'finetune':
                self.data = pickle.load(open('rawtest1data.pkl', "rb"))
                #self.data.sort(key=lambda x: len(x['rulelist']), reverse=True)
            else:
                self.data = pickle.load(open('fttest%d.pkl'%idx, 'rb'))

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
        if os.path.exists("typedict.pkl"):
            self.typedict = pickle.load(open("typedict.pkl", "rb"))
        if os.path.exists("edgedict.pkl"):
            self.edgedict = pickle.load(open("edgedict.pkl", "rb"))
        #self.Nl_Voc["<emptynode>"] = len(self.Nl_Voc)
        #self.Code_Voc["<emptynode>"] = len(self.Code_Voc)

    def init_dic(self):
        self.Code_Voc = {'pad':0}
        for x in self.ruledict:
            lst = x.strip().lower().split()
            if len(lst) < 2:
                continue
            tmp = [lst[0]] + lst[2:]
            for y in tmp:
                if y not in self.Code_Voc:
                    self.Code_Voc[y] = len(self.Code_Voc)
        assert("root" in self.Code_Voc)
        for x in self.Code_Voc:
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
        open("char_voc.pkl", "wb").write(pickle.dumps(self.Char_Voc))
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            x = x.lower()
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            x = x.lower()
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def processOne(self, s):
        inputNl = {'nl':tokenizer.encode(s)[1:-1], 'rulelist':[1,1]}
        self.data = [inputNl]
        return
    def gen_next(self, s2):
        k = -1
        n = len(s2)
        j = 0
        next_list = [0 for i in range(n)]
        next_list[0] = -1                           #next数组初始值为-1
        while j < n-1:
            if k == -1 or s2[k] == s2[j]:
                k += 1
                j += 1
                next_list[j] = k                    #如果相等 则next[j+1] = k
            else:
                k = next_list[k]                    #如果不等，则将next[k]的值给k
        return next_list


    def match(self, s1, s2):
        sc = s1 + "kk" + s2
        if sc in self.cache:
            #sc = s1 + "kk" + s2
            return self.cache[sc] / len(s2)
        #elif len(s2) >= len(s1) and s2 + "kk" + s1 in self.cache:
        #    sc = s2 + "kk" + s1
        #    return self.cache[s2 + "kk" + s1] / len(s2)
        #if len(s2) < len(s1):
        #    sc = s1 + "kk" + s2
        #else:
        #    sc = s2 + "kk" + s1
        ans = -1
        next_list = self.gen_next(s2)
        i = 0
        j = 0
        ma_len = 0
        while i < len(s1):
            if s1[i] == s2[j] or j == -1:
                i += 1
                j += 1
                ma_len = max(ma_len, j)
            else:
                j = next_list[j]
            if j == len(s2):
                ans = i - len(s2)
                break
        self.cache[sc] = ma_len
        return ma_len / len(s2)
    def getMaskByfather(self, child, father):
        maxad = 10
        admask = Graph()#np.zeros([self.max_len, self.max_len])
        demask = Graph()#np.zeros([self.max_len, self.max_len])
        for i in ((child)):
            for j in range(len(child[i])):
                for k in range(j + 1, len(child[i])):
                    admask.addEdge(child[i][k], child[i][j], min(k - j, self.max_rel_pos))#[k, j] = min(k -j, 10) + 1
                    assert(child[i][k] - child[i][j] != 0)
            tmpid = i
            depth = 0
            while tmpid != 0:
                tmpf = father[tmpid]
                depth += 1
                demask.addEdge(i, tmpf, min(depth, self.max_rel_pos))#demask[i, tmpf] = depth + 10
                tmpid = tmpf
        return admask, demask
    def preProcessData(self, dataFile):
        inputNl = []
        inputRuleParent = []
        inputRuleChild = []
        inputParent = []
        inputParentPath = []
        inputRes = []
        inputRule = []
        inputParentf = []
        inputParenta = []
        nls = []
        for data in tqdm(dataFile):
            child = {}
            father = {}
            #print(data['nl'])
            nl = tokenizer.encode(data['nl'].replace('concode_elem_sep', tokenizer.sep_token))#lines[5 * i].lower().strip().split()
            #nls.append(nl)
            inputparent = data['fatherlist']
            inputres = data['rulelist'][1:]#lines[5 * i + 1].strip().split()
            #print(inputres)
            #depth = lines[5 * i + 3].strip().split()
            #parentname = data['fathername']#lines[5 * i + 4].strip().lower().split()
            adrow = [0]
            frow = [0]
            inputrule = [self.ruledict["start -> java"]]
            inputad = Graph()
            #inputde = [0]#Graph()
            for j in range(len(inputres)):
                inputres[j] = int(inputres[j])
                inputparent[j] = int(inputparent[j]) + 1
                if inputres[j] >= 1000000:
                    assert(0)
                    inputres[j] = len(self.ruledict) + inputres[j] - 1000000
                    if inputres[j] - len(self.ruledict) >= len(nl):
                        print(nl)
                    assert(inputres[j] - len(self.ruledict) < len(nl))
                    if j + 1 < self.Code_Len:
                        adrow.append(self.Nl_Len + j + 1)
                        adcol.append(inputres[j] - len(self.ruledict))
                        adval.append(1)
                        #inputad[self.Nl_Len + j + 1, inputres[j] - len(self.ruledict)] = 1
                    inputrule.append(self.ruledict['start -> copyword'])
                else:
                    inputrule.append(inputres[j])
                if j + 1 < self.Code_Len:
                    if inputparent[j] in child:
                        #for y in child[inputparent[j]]:
                        #    inputad.addEdge(j + 1, y, 1)
                        adrow.append(child[inputparent[j]][-1])
                    else:
                        adrow.append(inputparent[j])
                child.setdefault(inputparent[j], []).append(j + 1)
                father[j + 1] = inputparent[j]
                if inputres[j] - len(self.ruledict) >= self.Nl_Len:
                    print(inputres[j] - len(self.ruledict))
                if j + 1 < self.Code_Len:
                    inputad.addEdge(j + 1, inputparent[j], 1)
                    frow.append(inputparent[j])
            admask, demask = self.getMaskByfather(child, father)
            #inputParentf.append(demask)
            #inputParenta.append(admask)
            inputNl.append(nl)
            inputrule = inputrule#self.pad_seq(inputrule, self.Code_Len)
            inputres = inputres#self.pad_seq(inputres, self.Code_Len)
            inputRes.append(inputres)
            inputRule.append(inputrule)
            inputad.normlize()
            inputParent.append(inputad)
            inputParentf.append(frow)
            inputParenta.append(adrow)
        batchs = [inputNl, inputRule, inputRes, inputParent, inputParentf, inputParenta]
        self.data = batchs
        #self.code = codes
        if self.dataName == "train":
            open("data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        if self.dataName == "val":
            open("valdata.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        if self.dataName == "test":
            open("testdata.pkl", "wb").write(pickle.dumps(batchs))
            open("testnl.pkl", "wb").write(pickle.dumps(nls))
        return batchs
    def __getitem__(self, offset):
        return self.data[offset]
    def __len__(self):
        return len(self.data)
