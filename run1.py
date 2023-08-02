import torch
from torch import optim
from Dataset import SumDataset,rs_collate_fn
import os
from tqdm import tqdm
from Model import *
import numpy as np
from copy import deepcopy
import pickle
import sys
from ScheduledOptim import *
from scipy import sparse
import json
from torchsummary import summary
from stringfycode import stringfyNode
from scheduler import PolynomialLRDecay
from transformers import AutoModel, AutoTokenizer
onelist = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers"]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
from torch import multiprocessing as mp
#sys.setrecursionlimit(500000000)
sys.setrecursionlimit(500000000)
#from pythonBottom.run import finetune
#from pythonBottom.run import pre
#wandb.init("sql")
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':512,
    'CodeLen':512,
    'batch_size':63,
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
})
os.environ["CUDA_VISIBLE_DEVICES"]="3, 0, 1, 2, 4, 5, 6, 7"
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
def save_model(model, dirs='checkpointSearch/', optimizer=None, amp=None):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    '''if args.use_apex:
        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'amp':amp.state_dict()
        }
        torch.save(checkpoint, dirs + 'best_model.ckpt')
    else:'''
    torch.save(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs = 'checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt'))
use_cuda = torch.cuda.is_available()
def gVar(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    else:
        assert isinstance(tensor, torch.Tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor
def getAntiMask(size):
    ans = np.zeros([size, size])
    for i in range(size):
        for j in range(0, i + 1):
            ans[i, j] = 1.0
    return ans
def getAdMask(size):
    ans = np.zeros([size, size])
    for i in range(size - 1):
        ans[i, i + 1] = 1.0
    return ans
def getRulePkl(vds):
    inputruleparent = []
    inputrulechild = []
    for i in range(args.cnum):
        rule = vds.rrdict[i].strip().lower().split()
        if len(rule) < 2:
            continue 
        inputrulechild.append(vds.pad_seq(vds.Get_Em(rule[2:], vds.Code_Voc), vds.Char_Len))
        inputruleparent.append(vds.Code_Voc[rule[0].lower()])
    return np.array(inputruleparent), np.array(inputrulechild)
def getAstPkl(vds):
    rrdict = {}
    for x in vds.Code_Voc:
        rrdict[vds.Code_Voc[x]] = x
    inputchar = []
    for i in range(len(vds.Code_Voc)):
        rule = rrdict[i].strip().lower()
        inputchar.append(vds.pad_seq(vds.Get_Char_Em([rule])[0], vds.Char_Len))
    return np.array(inputchar)
def train():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_set = SumDataset(args, "train")
    dev_set = SumDataset(args, "test")
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float()
    tmpast = getAstPkl(train_set)
    args.cnum = len(train_set.ruledict)
    #print(len(train_set.edgedict))
    print(args.bertnum)
    a, b = getRulePkl(train_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.batch_size, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex = gVar(np.arange(args.bertnum, len(train_set.ruledict))).unsqueeze(0).repeat(args.batch_size, 1).long() - args.bertnum
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(train_set.Code_Voc))).unsqueeze(0).repeat(args.batch_size, 1).long()
    args.Code_Vocsize = len(train_set.Code_Voc)
    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict)
    #dev_set = SumDataset(args, "val")
    args.edgelen = len(train_set.edgedict)
    model = Decoder(args)
    load_model(model, 'checkModelNUM/')
    '''for n, x in model.named_parameters():
        print(n, x.numel())
    total_params3 = sum(x.numel() for x in model.parameters())
    total_params = sum(p.numel() for p in model.encoder.parameters())
    total_params1 = sum(p.numel() for p in model.encodeTransformerBlock.parameters())
    total_params2 = sum(p.numel() for p in model.decodeTransformerBlocksP.parameters())

    print(f'{total_params:,} total parameters.')
    print(f'{total_params1:,} total parameters in encoder.')
    print(f'{total_params2:,} total parameters in decoder.')
    print(f'{total_params3:,} total parameters in model.')'''

    #nlem = pickle.load(open("embedding.pkl", "rb"))
    #model.encoder.token_embedding.token.em.weight.data.copy_(gVar(nlem))
    #charem = pickle.load(open("char_embedding.pkl", "rb"))
    #model.encoder.char_embedding.token.em.weight.data.copy_(gVar(charem))
    #model.ad = rulead
    #load_model(model)
    base_params = list(map(id, model.encoder.model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": 1e-4, 'notchange':False},
        {"params": model.encoder.model.parameters(), "lr": 5e-5, 'notchange':True},
    ]
    optimizer = optim.AdamW(params, eps=1e-8)
    #scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=100000, init_lr0=3e-4, init_lr2=5e-5, end_learning_rate=0.000, power=1.0)
    pathnames = []
    num_trial = patience = 0
    isBetter = False
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, max_steps=50000)
    maxAcc= 0
    maxC = 0
    minloss = 1e10
    if use_cuda:
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
        torch.cuda.manual_seed_all(args.seed)
        from blcDP import BalancedDataParallel
        model = BalancedDataParallel(0, model)#nn.DataParallel(model, device_ids=[0, 1])
    #model.to()
    for epoch in range(100000):
        j = 0
        

        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True, num_workers=4,collate_fn=rs_collate_fn)
        for dBatch in tqdm(data_loader):
            isBetter = False
            if j % 400 == 0:
                devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=True, num_workers=4,collate_fn=rs_collate_fn)
                model = model.eval()
                accs = []
                tcard = []
                tmp = []
                rulead2 = rulead.unsqueeze(0).repeat(args.batch_size, 1, 1)
                for devBatch in tqdm(devloader):
                    for i in range(len(devBatch)):
                        devBatch[i] = gVar(devBatch[i])
                    antimask = gVar(getAntiMask(devBatch[1].size(1)))
                    antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)

                    with torch.no_grad():
                        bsize = devBatch[0].size(0)
                        #print(bsize)
                        tmpindex = tmpindex[:bsize]
                        tmpf = tmpf[:bsize]
                        tmpc = tmpc[:bsize]
                        tmpchar = tmpchar[:bsize]
                        tmpindex2 = tmpindex2[:bsize]
                        rulead2 = rulead2[:bsize]
                        antimask2 = antimask2[:bsize]
                        _, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[4], devBatch[5], tmpindex, tmpf, tmpc, tmpchar, tmpindex2, rulead2, antimask2, devBatch[3])
                        #print(pre.size())
                        pred = pre.argmax(dim=-1)
                        resmask = torch.ne(devBatch[3], args.mask_id)
                        acc = (torch.eq(pred, devBatch[3]) * resmask).float()#.mean(dim=-1)
                        predres = (1 - acc) * pred.float() * resmask.float()
                        accsum = torch.sum(acc, dim=-1)
                        resTruelen = torch.sum(resmask, dim=-1).float()
                        cnum = (torch.eq(accsum, resTruelen)).sum().float()
                        acc = acc.sum(dim=-1) / resTruelen
                        accs.append(acc.mean().item())
                        tcard.append(cnum.item())
                        #print(devBatch[5])
                        #print(predres)
                tnum = np.sum(tcard)
                acc = np.mean(accs)
                #wandb.log({"accuracy":acc})
                print(str(acc), str(tnum), str(maxC))
                #print(tmp)
                if maxC < tnum:
                    maxC = tnum
                    save_model(model.module, 'checkModelNUM/')
                    isBetter = True
                    print('find better accuracy %f'%tnum)
                    #save_model(model)
                if maxAcc < acc:
                    isBetter = True
                    maxAcc = acc
                if isBetter:
                    patience = 0
                    print('save model to [%s]' % 'checkModel/', file=sys.stderr)
                    #save_model(model.module, 'checkModel%d-%d/'%(epoch, j))
                    save_model(model.module, 'checkModel/')
                    torch.save(optimizer.state_dict(), 'checkModel/optim.bin')

                elif patience < args.patience:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    if patience == args.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == args.max_num_trials:
                            print('early stop!', file=sys.stderr)
                            exit(0)
                        lr = optimizer.param_groups[0]['lr'] * 0.5
                        model.module.load_state_dict(torch.load('checkModel/best_model.ckpt'))
                        model = model.cuda()
                        optimizer.load_state_dict(torch.load('checkModel/optim.bin'))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0
                else:
                    patience += 1
            rulead2 = rulead.unsqueeze(0).repeat(args.batch_size, 1, 1)
            model = model.train()
            for i in range(len(dBatch)):
                dBatch[i] = gVar(dBatch[i])
            antimask = gVar(getAntiMask(dBatch[1].size(1)))
            antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)
            loss, _ = model(dBatch[0], dBatch[1], dBatch[2], dBatch[4], dBatch[5], tmpindex, tmpf, tmpc, tmpchar, tmpindex2, rulead2, antimask2, dBatch[3])
            #print(loss.data.cpu().numpy())
            #loss = torch.mean(loss)
            '''if loss.item() == np.inf:
                rrdict = {}
                for x in dev_set.Nl_Voc:
                    rrdict[dev_set.Nl_Voc[x]] = x
                ans = ""
                for x in dBatch[0][0].data.cpu().numpy():
                    ans += rrdict[x]
                print(ans)
                #for i in range(len(dBatch)):
                    #print(dBatch[i][16].data.cpu().numpy())
                exit(0)'''
            loss = torch.mean(loss)#+ F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 2, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 3, 1).squeeze(0).squeeze(0).mean() + F.max_pool1d(loss.unsqueeze(0).unsqueeze(0), 4, 1).squeeze(0).squeeze(0).mean()
            if loss.item() == np.inf:
                print(j)
                assert(0)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()
            if j % args.gradient_accumulation_steps == 0:
                optimizer.step()#_and_update_lr()
                optimizer.zero_grad()
            #scheduler_poly_lr_decay.step()
            j += 1
def mergeIdentifier(root):
    if 'identifier' in root.name or 'literal' in root.name:
        if False:
            pass
        else:
            oname = ""
            for x in root.child:
                oname += x.name[:-4]
            oname += "_ter"
            nnode = Node(oname, root.depth)
            nnode.father = root
            root.child = [nnode]
    for x in root.child:
        mergeIdentifier(x)
    return
class Node:
    def __init__(self, name, d):
        self.name = name
        self.depth = d
        self.father = None
        self.child = []
        self.sibiling = None
        self.expanded = False
        self.fatherlistID = 0
class SearchNode:
    def __init__(self, ds, nl):
        self.state = [ds.ruledict["start -> root"]]
        self.prob = 0
        self.aprob = 0
        self.bprob = 0
        self.finish = False
        self.root = Node("method_declaration", 2)
        self.inputparent = ["start"]
        self.parentrow = [] #= np.zeros([args.NlLen + args.CodeLen + args.TableLen, args.NlLen + args.CodeLen + args.TableLen])
        self.parentcol = []
        self.parentdata = []
        #self.parent[args.NlLen]
        self.expanded = None
        #self.ruledict = ds.rrdict
        self.expandedname = []
        self.depth = [ds.pad_seq([1], 40)]
        self.child = {}
        self.idmap = {}
        for x in ds.ruledict:
            self.expandedname.append(x.strip().split()[0])
        self.expandedname.extend(onelist)
        self.everTreepath = []
    def selcetNode(self, root):
        if not root.expanded and (root.name in self.expandedname) and root.name not in onelist and ("identifier" not in root.name and 'literal' not in root.name):
            return root
        else:
            for x in root.child:
                ans = self.selcetNode(x)
                if ans:
                    return ans
            if (root.name in onelist or ("identifier" in root.name or 'literal' in root.name)) and root.expanded == False:
                return root
        return None
    def selectExpandedNode(self):
        self.expanded = self.selcetNode(self.root)
    def getRuleEmbedding(self, ds):
        inputruleparent = []
        inputrulechild = []
        for x in self.state:
            if x >= len(ds.rrdict):
                inputruleparent.append(ds.Get_Em(["value"], ds.Code_Voc)[0])
                inputrulechild.append(ds.pad_seq(ds.Get_Em(["copyword"], ds.Code_Voc), ds.Char_Len))
            else:
                rule = ds.rrdict[x].strip().lower().split()
                inputruleparent.append(ds.Get_Em([rule[0]], ds.Code_Voc)[0])
                inputrulechild.append(ds.pad_seq(ds.Get_Em(rule[2:], ds.Code_Voc), ds.Char_Len))
        tmp = [ds.pad_seq(ds.Get_Em(['start'], ds.Code_Voc), 10)] + self.everTreepath        
        inputrule = ds.pad_seq(self.state, ds.Code_Len)
        inputrulechild = ds.pad_list(tmp, ds.Code_Len, 10)
        #inputrulechild = ds.pad_list(inputrulechild, ds.Code_Len, ds.Char_Len)
        inputruleparent = ds.pad_seq(ds.Get_Em(self.inputparent, ds.Code_Voc), ds.Code_Len)
        inputdepth = ds.pad_list(self.depth, ds.Code_Len, 40)
        #inputdepth = ds.pad_seq(self.depth, ds.Code_Len)
        return inputrule, inputrulechild, inputruleparent, inputdepth
    def getTreePath(self, ds):
        tmppath = [self.expanded.name.lower()]
        node = self.expanded.father
        while node:
            tmppath.append(node.name.lower())
            node = node.father
        tmp = ds.pad_seq(ds.Get_Em(tmppath, ds.Code_Voc), 10)
        self.everTreepath.append(tmp)
        return ds.pad_list(self.everTreepath, ds.Code_Len, 10)
    def copynode(self, newnode, original):
        for x in original.child:
            nnode = Node(x.name, 0)
            nnode.father = newnode
            nnode.expanded = True
            newnode.child.append(nnode)
            self.copynode(nnode, x)
        return
    def checkapply(self, rule, ds):
        rules = ds.rrdict[rule]
        lst = rules.strip().split()
        if "->" not in rules or lst[0] == '->':
            if lst[0] == '->' and self.expanded.name != 'string_literal':
                return False
            else:
                if ("identifier" not in self.expanded.name and 'literal' not in self.expanded.name):
                    return False
        else:
            rules = ds.rrdict[rule]
            if rules.strip().split()[0].lower() != self.expanded.name.lower():
                return False
        return True
    def applyrule(self, rule, ds):
        #print(rule)
        #print(self.state)
        #print(self.printTree(self.root))
        rules = ds.rrdict[rule]
        lst = rules.strip().split()
        if "->" not in rules or lst[0] == '->':
            if rules == ' -> String_ter ':
                nnode = Node("srini_string_ter", 0)
            else:
                nnode = Node(lst[0] + '_ter', 0)
            nnode.father = self.expanded
            nnode.fatherlistID = len(self.state)
            self.expanded.child.append(nnode)
        else:
            rules = ds.rrdict[rule]
            #print(rules)
            if rules.strip().split()[0].lower() != self.expanded.name.lower():
                assert(0)
                return False
            #assert(rules.strip().split()[0] == self.expanded.name)
            if rules == self.expanded.name + " -> End ":
                self.expanded.expanded = True
            else:
                for x in rules.strip().split()[2:]:
                    if self.expanded.depth + 1 >= 40:
                        nnode = Node(x, 39)
                    else:
                        nnode = Node(x, self.expanded.depth + 1)                   
                    #nnode = Node(x, self.expanded.depth + 1)
                    self.expanded.child.append(nnode)
                    nnode.father = self.expanded
                    nnode.fatherlistID = len(self.state)
        #self.parent.append(self.expanded.fatherlistID)
        self.parentrow.append(len(self.state))
        self.parentcol.append(self.expanded.fatherlistID)
        #print('-', self.expanded.fatherlistID)
        assert(self.expanded.fatherlistID != -1)
        self.parentdata.append(1)
        #self.parent[args.NlLen + len(self.depth), args.NlLen + self.expanded.fatherlistID] = 1
        if rule >= len(ds.ruledict):
            assert(0)
            self.parentrow.append(args.NlLen + args.TableLen + len(self.depth))
            if rule - len(ds.ruledict) >= args.CodeLen: 
                self.parentcol.append(rule - len(ds.ruledict) - args.CodeLen + args.NlLen)
            else:
                self.parentcol.append(rule - len(ds.ruledict) + args.NlLen + args.TableLen)
            self.parentdata.append(1)
            #self.parent[args.NlLen + args.TableLen + len(self.depth), rule - len(ds.ruledict)] = 1
        if rule >= len(ds.ruledict):
            assert(0)
            if rule - len(ds.ruledict) >= args.CodeLen: 
                if self.expanded.name == 'T':
                    self.state.append(ds.ruledict['start -> tab'])
                if self.expanded.name == 'CR':
                    self.state.append(ds.ruledict['start -> col'])
            else:
                self.state.append(ds.ruledict['start -> copyword'])
        else:
            self.state.append(rule)
        self.inputparent.append(self.expanded.name.lower())
        tmpdepth = []
        if self.expanded.father:
            if self.expanded.fatherlistID not in self.child:
                ids = 1
            else:
                ids = len(self.child[self.expanded.fatherlistID]) + 1
            self.child.setdefault(self.expanded.fatherlistID, []).append(len(self.state) - 1)
        else:
            ids = 1
        tmpdepth.append(ids)
        tmpdepth.extend([])
        tmpdepth = ds.pad_seq(tmpdepth, 40)
        #self.depth.append(tmpdepth)
        #self.depth.append(self.expanded.depth)
        if self.expanded.name not in onelist:
            self.expanded.expanded = True
        if ("identifier" in self.expanded.name or 'literal' in self.expanded.name): #self.expanded.name in ['qualifier', 'member', 'name', 'value', 'flag']:
            if 'Ġ' in rules:
                self.expanded.child.reverse()
                self.expanded.expanded = True
            else:
                self.expanded.expanded = False
        return True
    def printTree(self, r):
        s = r.name + " "#print(r.name)
        if len(r.child) == 0:
            s += "^ "
            return s
        #r.child = sorted(r.child, key=lambda x:x.name)
        for c in r.child:
            s += self.printTree(c)
        s += "^ "#print(r.name + "^")
        return s
    def getTreestr(self):
        return self.printTree(self.root)

        
beamss = []
def isvalid(root, table):
    valid = True
    if root.name == 'CR' and root.father.child[1].name == 'T' and root.father.child[1].child[0].name != 'Root':
        if root.child[0].name == 'TIME_NOW':
            return True
        #print(table)
        colid = int(root.child[0].name) - len(table['table_names'])
        if colid == 0:
            return True
        tableid = int(root.father.child[1].child[0].name)
        if colid >= len(table['column_names']):
            print('test1')
            #print(colid, len(table['column_names']))
            return False
        if table['column_names'][colid][0] == tableid:
            return True
        else:
            #print(colid, tableid)
            #print('test2')
            #assert(0)
            #print(len(table['table_names']), colid, table['column_names'][colid][0], tableid)
            return False
    for x in root.child:
        valid = valid and isvalid(x, table)
    return valid

def BeamSearch(inputnl, vds, model, beamsize, batch_size, k, queue=None):
    print(inputnl[0].shape)
    batch_size = gVar(inputnl[0]).size(0)
    #batch_size = args.batch_size
    rulead2 = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(args.batch_size, 1, 1)
    a, b = getRulePkl(vds)
    tmpf = gVar(a).unsqueeze(0).repeat(args.batch_size, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex = gVar(np.arange(args.bertnum, len(vds.ruledict))).unsqueeze(0).repeat(args.batch_size, 1).long() - args.bertnum
    tmpast = getAstPkl(vds)
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(vds.Code_Voc))).unsqueeze(0).repeat(args.batch_size, 1).long()
    args.NlLen = inputnl[0].shape[1]
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            beams[i] = [SearchNode(vds, [])]
        nlencode, nlmask = model.nl_encode(gVar(inputnl[0]))
        index = 0
        endnum = {}
        continueSet = {}
        while True:
            #print(index)
            args.CodeLen = min(index + 2, 512)
            vds.Code_Len = min(index + 2, 512)#index + 1
            antimask = gVar(getAntiMask(args.CodeLen))
            tmpbeam = {}
            ansV = {}
            if len(endnum) == batch_size:
                break
            if index >= args.CodeLen:
                break
            for p in range(beamsize):
                tmprule = []
                tmprulechild = []
                tmpruleparent = []
                tmptreepath = []
                tmpAd = []
                validnum = []
                tmpdepth = []
                for i in range(batch_size):
                    if p >= len(beams[i]):
                        continue
                    x = beams[i][p]
                    if not x.finish:
                        x.selectExpandedNode()
                    if x.expanded == None or len(x.state) >= args.CodeLen:
                        x.finish = True
                        ansV.setdefault(i, []).append(x)
                    else:
                        #print(x.expanded.name)
                        validnum.append(i)
                        a, b, c, d = x.getRuleEmbedding(vds)
                        tmprule.append(a)
                        tmprulechild.append(b)
                        tmpruleparent.append(c)
                        tmptreepath.append(x.getTreePath(vds))
                        #tmp = np.eye(vds.Code_Len)[x.parent]
                        #tmp = np.concatenate([tmp, np.zeros([vds.Code_Len, vds.Code_Len])], axis=0)[:vds.Code_Len,:]#self.pad_list(tmp, self.Code_Len, self.Code_Len)
                        parent = sparse.coo_matrix((x.parentdata, (x.parentrow, x.parentcol)), shape=[args.CodeLen, args.CodeLen])
                        tmpAd.append(parent.toarray())
                #print("--------------------------")
                if len(tmprule) == 0:
                    continue
                batch_sizess = len(tmprule)
                antimask2 = antimask.unsqueeze(0).repeat(batch_sizess, 1, 1).unsqueeze(1)
                tmprule = np.array(tmprule)
                tmprulechild = np.array(tmprulechild)
                tmpruleparent = np.array(tmpruleparent)
                tmptreepath = np.array(tmptreepath)
                tmpAd = np.array(tmpAd)
                bsize = batch_sizess
                tmpindex = tmpindex[:bsize]
                tmpf = tmpf[:bsize]
                tmpc = tmpc[:bsize]
                tmpchar = tmpchar[:bsize]
                tmpindex2 = tmpindex2[:bsize]
                rulead2 = rulead2[:bsize]
                antimask2 = antimask2[:bsize]
                #print(gVar(inputnl[0][validnum]).view(-1, args.NlLen))
                result = model.test_foward(nlencode[validnum], nlmask[validnum], gVar(tmprule), gVar(tmprulechild), gVar(tmpAd), gVar(tmptreepath), tmpindex, tmpf, tmpc, tmpchar, tmpindex2, rulead2, antimask2, None, "test")
                results = result#indexs = torch.argsort(result, descending=True)#results = result.data.cpu().numpy()
                currIndex = 0
                for j in range(batch_size):
                    if j not in validnum:
                        continue
                    x = beams[j][p]
                    tmpbeamsize = 0
                    result = results[currIndex, index]#np.negative(results[currIndex, index])
                    currIndex += 1
                    cresult = result#np.negative(result)
                    indexs = torch.argsort(result, descending=True)
                    for i in range(len(indexs)):
                        if tmpbeamsize >= 30:
                            break
                        #copynode = pickle.loads(pickle.dumps(x))#deepcopy(x)
                        #if indexs[i] >= len(vds.rrdict):
                            #print(cresult[indexs[i]])
                        #print('-', indexs[i])
                        c = x.checkapply(indexs[i].item(), vds)
                        #c = copynode.applyrule(indexs[i], vds.nl[args.batch_size * k + j], vds.tabless[args.batch_size * k + j], vds)
                        if not c:
                            tmpbeamsize += 1
                            continue
                        prob = x.prob + np.log(cresult[indexs[i]].item())#copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append([prob, indexs[i].item(), x])#tmpbeam.setdefault(j, []).append(copynode)
            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
            for j in range(args.batch_size):
                if j in tmpbeam:
                    if j in ansV:
                        for x in ansV[j]:
                            tmpbeam[j].append([x.prob, -1, x])
                    tmp = sorted(tmpbeam[j], key=lambda x: x[0], reverse=True)[:beamsize]
                    beams[j] = []
                    for x in tmp:
                        if x[1] != -1:
                            copynode = pickle.loads(pickle.dumps(x[2]))
                            copynode.applyrule(x[1], vds)
                            copynode.prob = x[0]
                            beams[j].append(copynode)
                        else:
                            beams[j].append(x[2])
            index += 1
            

        for i in range(len(beams)):
            mans = -1000000
            lst = beams[i]
            tmpans = 0
            for y in lst:
                #f.write("\t".join(vds.nl[args.batch_size * k + i]) + "\n")
                #f.write(y.getTreestr() + "\n")
                #f.flush()
                mergeIdentifier(y.root)
                #print(y.getTreestr())
                #print(stringfy(y.root))
                if y.prob > mans:
                    mans = y.prob
                    tmpans = y
            beams[i] = tmpans
        if queue is not None:
            queue.put({'id':index, 'ans':beams})
        else:
            return beams
        #return beams
def stringfy(node):
    ans = ""
    if len(node.child) == 0:
        if node.name[0] == 'Ġ':
            ans += node.name[1:-4]
        else:
            ans = node.name[:-4]
    else:
        for x in node.child:
            ans += stringfy(x) + " "
    return ans
def testdis():
    #pre()
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    dev_set = SumDataset(args, "test")
    #rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float()
    #tmpast = getAstPkl(dev_set)
    args.cnum = len(dev_set.ruledict)
    #print(len(train_set.edgedict))
    print(args.bertnum)
    #a, b = getRulePkl(dev_set)
    #tmpf = gVar(a).unsqueeze(0).repeat(args.batch_size, 1).long()
    #tmpc = gVar(b).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    #tmpindex = gVar(np.arange(args.bertnum, len(dev_set.ruledict))).unsqueeze(0).repeat(args.batch_size, 1).long() - args.bertnum
    #tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    #tmpindex2 = gVar(np.arange(len(dev_set.Code_Voc))).unsqueeze(0).repeat(args.batch_size, 1).long()
    args.Code_Vocsize = len(dev_set.Code_Voc)
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    args.rulenum = len(dev_set.ruledict)
    #dev_set = SumDataset(args, "val")
    args.edgelen = len(dev_set.edgedict)
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    #print(dev_set.Nl_Voc)
    model = Decoder(args)

    load_model(model, 'checkModel/')
    if use_cuda:#torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        #model = model.cuda()
        model.to(0)
        #model = nn.DataParallel(model, device_ids=[0, 1])
    
    #model = model.module
    model.share_memory()
    ctx = mp.get_context('spawn')
    '''for n, x in model.named_parameters():
        print(n, x.numel())
    total_params3 = sum(x.numel() for x in model.parameters())
    total_params = sum(p.numel() for p in model.encoder.parameters())
    total_params1 = sum(p.numel() for p in model.encodeTransformerBlock.parameters())
    total_params2 = sum(p.numel() for p in model.decodeTransformerBlocksP.parameters())

    print(f'{total_params:,} total parameters.')
    print(f'{total_params1:,} total parameters in encoder.')
    print(f'{total_params2:,} total parameters in decoder.')
    print(f'{total_params3:,} total parameters in model.')
    assert(0)'''
    args.batch_size = 1
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0, collate_fn=rs_collate_fn)
    model = model.eval()
    #load_model(model)
    index = 0 
    #antimask = gVar(getAntiMask(args.CodeLen))
    #antimask2 = antimask.unsqueeze(0).repeat(1, 1, 1).unsqueeze(1)
    #rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    #a, b = getRulePkl(dev_set)
    #tmpf = gVar(a).unsqueeze(0).repeat(2, 1).long()
    #tmpc = gVar(b).unsqueeze(0).repeat(2, 1, 1).long()
    #tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).repeat(2, 1).long()
    res = ctx.SimpleQueue()
    process = []
    for x in tqdm(devloader):
        '''pre = model(gVar(x[0]), gVar(x[1]), gVar(x[2]), gVar(x[3]), gVar(x[4]), gVar(x[6]), gVar(x[7]), gVar(x[8]), gVar(x[9]), gVar(x[10]), gVar(x[11]), tmpindex, tmpf, tmpc, rulead, antimask2, None, 'test')
        #print(pre[0,3,4020], pre[0,3,317])
        pred = pre.argmax(dim=-1)
        resmask = torch.gt(gVar(x[5]), 0)
        acc = (torch.eq(pred, gVar(x[5])) * resmask).float()#.mean(dim=-1)
        predres = (1 - acc) * pred.float() * resmask.float()
        accsum = torch.sum(acc, dim=-1)
        resTruelen = torch.sum(resmask, dim=-1).float()
        cnum = (torch.eq(accsum, resTruelen)).sum().float()
        if cnum.item() != 1:
            index += 1
            continue'''
        
        if index <= -1:
            index += 1
            continue
        print(dev_set.nl[index])
        while True:
            activateprocess = 0
            for p in process:
                if p.poll() is not None:
                    process.remove(p)
                else:
                    activateprocess += 1
            if activateprocess < 10:
                break
        p = ctx.Process(target=BeamSearch, args=((x[0], x[1]), dev_set, model, 4, args.batch_size, index, res), name='process-%d'%(index))
        p.start()
        #ans = BeamSearch((x[0], x[1]), dev_set, model, 4, args.batch_size, index)
        index += 1
        while True:
            indexss = res.get()
            if indexss is None:
                break
            ans = indexss['ans']
            idx = indexss['id']
            f = open("result/%d.txt"%(idx), "w")
            for i in range(len(ans)):
                beam = ans[i]
                #print(beam[0].parent, beam[0].everTreepath, beam[0].state)
                f.write(beam.getTreestr())
                f.write("\n")
                f.write(stringfy(beam.root))
                f.write("\n")
                f.write(str(beam.state) + "\n")
                f.flush()   
            f.close()
        #exit(0)
        #f.write(" ".join(ans.ans[1:-1]))
        #f.write("\n")
        #f.flush()#print(ans)
    #open("beams1.pkl", "wb").write(pickle.dumps(beamss))
def eval():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev_set = SumDataset(args, "test")
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float()
    tmpast = getAstPkl(dev_set)
    args.cnum = len(dev_set.ruledict)
    #print(len(train_set.edgedict))
    print(args.bertnum)
    print(args.mask_id)
    a, b = getRulePkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.batch_size, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex = gVar(np.arange(args.bertnum, len(dev_set.ruledict))).unsqueeze(0).repeat(args.batch_size, 1).long() - args.bertnum
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(dev_set.Code_Voc))).unsqueeze(0).repeat(args.batch_size, 1).long()
    args.Code_Vocsize = len(dev_set.Code_Voc)
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    args.rulenum = len(dev_set.ruledict)
    #dev_set = SumDataset(args, "val")
    args.edgelen = len(dev_set.edgedict)
    model = Decoder(args)
    #nlem = pickle.load(open("embedding.pkl", "rb"))
    #model.encoder.token_embedding.token.em.weight.data.copy_(gVar(nlem))
    #charem = pickle.load(open("char_embedding.pkl", "rb"))
    #model.encoder.char_embedding.token.em.weight.data.copy_(gVar(charem))
    #model.ad = rulead
    #load_model(model)
    base_params = list(map(id, model.encoder.model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": 1e-4, 'notchange':False},
        {"params": model.encoder.model.parameters(), "lr": 5e-5, 'notchange':True},
    ]
    optimizer = optim.AdamW(params, eps=1e-8)
    #scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=100000, init_lr0=3e-4, init_lr2=5e-5, end_learning_rate=0.000, power=1.0)
    pathnames = []
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, max_steps=50000)
    maxAcc= 0
    maxC = 0
    minloss = 1e10
    if use_cuda:
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
        torch.cuda.manual_seed_all(args.seed)
        from blcDP import BalancedDataParallel
        model = nn.DataParallel(model, device_ids=[0, 1])#BalancedDataParallel(0, model)#nn.DataParallel(model, device_ids=[0, 1])
    antimask = gVar(getAntiMask(args.CodeLen))
    load_model(model)#model.to()
    index = []
    for epoch in range(1):
        j = 0
        if True:
            if True:
                devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False, num_workers=1,collate_fn=rs_collate_fn)
                model = model.eval()
                accs = []
                tcard = []
                tmp = []
                rulead2 = rulead.unsqueeze(0).repeat(args.batch_size, 1, 1)
                for devBatch in tqdm(devloader):
                    for i in range(len(devBatch)):
                        devBatch[i] = gVar(devBatch[i])
                    with torch.no_grad():
                        antimask = gVar(getAntiMask(devBatch[1].size(1)))
                        antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)

                        bsize = devBatch[0].size(0)
                        #print(bsize)
                        tmpindex = tmpindex[:bsize]
                        tmpf = tmpf[:bsize]
                        tmpc = tmpc[:bsize]
                        tmpchar = tmpchar[:bsize]
                        tmpindex2 = tmpindex2[:bsize]
                        rulead2 = rulead2[:bsize]
                        antimask2 = antimask2[:bsize]
                        _, pre = model(devBatch[0], devBatch[1], devBatch[2], devBatch[4], devBatch[5], tmpindex, tmpf, tmpc, tmpchar, tmpindex2, rulead2, antimask2, devBatch[3])
                        #print(pre.size())
                        pred = pre.argmax(dim=-1)
                        resmask = torch.ne(devBatch[3], args.mask_id)
                        acc = (torch.eq(pred, devBatch[3]) * resmask).float()#.mean(dim=-1)
                        predres = (1 - acc) * pred.float() * resmask.float()
                        accsum = torch.sum(acc, dim=-1)
                        resTruelen = torch.sum(resmask, dim=-1).float()
                        cnum = (torch.eq(accsum, resTruelen)).sum().float()
                        acc = acc.sum(dim=-1) / resTruelen
                        accs.append(acc.mean().item())
                        tcard.append(cnum.item())
                        #print(devBatch[5])
                        if True:
                            '''for example in range(bsize):
                                for k in range(int(resTruelen[example].item())):
                                    print(dev_set.rrdict[devBatch[3][example][k].item()], end=" ")
                                    print(dev_set.rrdict[pred[example][k].item()], end=" ")
                                    print()'''
                            for k, x in enumerate(torch.eq(accsum, resTruelen)):
                                if x.item() == 1:
                                    index.append(k + args.batch_size * j)
                            print(torch.eq(accsum, resTruelen))
                        j += 1
                        #print(predres)
                tnum = np.sum(tcard)
                acc = np.mean(accs)
                
                print(index)
                #wandb.log({"accuracy":acc})
                print(str(acc), str(tnum), str(maxC))
def testone():
    #pre()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev_set = SumDataset(args, "testone")
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float()
    tmpast = getAstPkl(dev_set)
    args.cnum = len(dev_set.ruledict)
    #print(len(train_set.edgedict))
    print(args.bertnum)
    print(args.mask_id)
    a, b = getRulePkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.batch_size, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex = gVar(np.arange(args.bertnum, len(dev_set.ruledict))).unsqueeze(0).repeat(args.batch_size, 1).long() - args.bertnum
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(dev_set.Code_Voc))).unsqueeze(0).repeat(args.batch_size, 1).long()
    args.Code_Vocsize = len(dev_set.Code_Voc)
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    args.rulenum = len(dev_set.ruledict)
    #dev_set = SumDataset(args, "val")
    args.edgelen = len(dev_set.edgedict)
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    #print(dev_set.Nl_Voc)
    model = Decoder(args)
    s = "sort the array"
    dev_set.processOne(s)
    load_model(model, "checkModel/")
    if use_cuda:#torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
        #model = nn.DataParallel(model, device_ids=[0, 1])
    
    model = model
    args.batch_size = 1
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0, collate_fn=rs_collate_fn)
    model = model.eval()
    #load_model(model)
    f = open("outval1.txt", "w")
    index = 0 
    antimask = gVar(getAntiMask(args.CodeLen))
    antimask2 = antimask.unsqueeze(0).repeat(1, 1, 1).unsqueeze(1)
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    a, b = getRulePkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(2, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(2, 1, 1).long()
    tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).repeat(2, 1).long()
    for x in tqdm(devloader):
        '''pre = model(gVar(x[0]), gVar(x[1]), gVar(x[2]), gVar(x[3]), gVar(x[4]), gVar(x[6]), gVar(x[7]), gVar(x[8]), gVar(x[9]), gVar(x[10]), gVar(x[11]), tmpindex, tmpf, tmpc, rulead, antimask2, None, 'test')
        #print(pre[0,3,4020], pre[0,3,317])
        pred = pre.argmax(dim=-1)
        resmask = torch.gt(gVar(x[5]), 0)
        acc = (torch.eq(pred, gVar(x[5])) * resmask).float()#.mean(dim=-1)
        predres = (1 - acc) * pred.float() * resmask.float()
        accsum = torch.sum(acc, dim=-1)
        resTruelen = torch.sum(resmask, dim=-1).float()
        cnum = (torch.eq(accsum, resTruelen)).sum().float()
        if cnum.item() != 1:
            index += 1
            continue'''
        
        #print(dev_set.nl[index], x[0])
        ans = BeamSearch((x[0], 0), dev_set, model, 5, args.batch_size, index)
        index += 1
        for i in range(len(ans)):
            beam = ans[i]
            #print(beam[0].parent, beam[0].everTreepath, beam[0].state)
            f.write(beam.getTreestr())
            f.write("\n")
            f.write(str(beam.state) + "\n")
        #exit(0)
        #f.write(" ".join(ans.ans[1:-1]))
        #f.write("\n")
        #f.flush()#print(ans)
    open("beams1.pkl", "wb").write(pickle.dumps(beamss))
def test():
    #pre()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev_set = SumDataset(args, "test")
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float()
    tmpast = getAstPkl(dev_set)
    args.cnum = len(dev_set.ruledict)
    #print(len(train_set.edgedict))
    print(args.bertnum)
    a, b = getRulePkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(args.batch_size, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex = gVar(np.arange(args.bertnum, len(dev_set.ruledict))).unsqueeze(0).repeat(args.batch_size, 1).long() - args.bertnum
    tmpchar = gVar(tmpast).unsqueeze(0).repeat(args.batch_size, 1, 1).long()
    tmpindex2 = gVar(np.arange(len(dev_set.Code_Voc))).unsqueeze(0).repeat(args.batch_size, 1).long()
    args.Code_Vocsize = len(dev_set.Code_Voc)
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    args.rulenum = len(dev_set.ruledict)
    #dev_set = SumDataset(args, "val")
    args.edgelen = len(dev_set.edgedict)
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    #print(dev_set.Nl_Voc)
    model = Decoder(args)

    load_model(model, 'checkModel/')
    if use_cuda:#torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1])
    
    model = model.module
    ctx = mp.get_context('spawn')
    '''for n, x in model.named_parameters():
        print(n, x.numel())
    total_params3 = sum(x.numel() for x in model.parameters())
    total_params = sum(p.numel() for p in model.encoder.parameters())
    total_params1 = sum(p.numel() for p in model.encodeTransformerBlock.parameters())
    total_params2 = sum(p.numel() for p in model.decodeTransformerBlocksP.parameters())

    print(f'{total_params:,} total parameters.')
    print(f'{total_params1:,} total parameters in encoder.')
    print(f'{total_params2:,} total parameters in decoder.')
    print(f'{total_params3:,} total parameters in model.')
    assert(0)'''
    args.batch_size = 10
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0, collate_fn=rs_collate_fn)
    model = model.eval()
    #load_model(model)
    f = open("outval1.txt", "w")
    index = 0 
    antimask = gVar(getAntiMask(args.CodeLen))
    antimask2 = antimask.unsqueeze(0).repeat(1, 1, 1).unsqueeze(1)
    rulead = gVar(pickle.load(open("rulead.pkl", "rb"))).float().unsqueeze(0).repeat(2, 1, 1)
    a, b = getRulePkl(dev_set)
    tmpf = gVar(a).unsqueeze(0).repeat(2, 1).long()
    tmpc = gVar(b).unsqueeze(0).repeat(2, 1, 1).long()
    tmpindex = gVar(np.arange(len(dev_set.ruledict))).unsqueeze(0).repeat(2, 1).long()
    for x in tqdm(devloader):
        '''pre = model(gVar(x[0]), gVar(x[1]), gVar(x[2]), gVar(x[3]), gVar(x[4]), gVar(x[6]), gVar(x[7]), gVar(x[8]), gVar(x[9]), gVar(x[10]), gVar(x[11]), tmpindex, tmpf, tmpc, rulead, antimask2, None, 'test')
        #print(pre[0,3,4020], pre[0,3,317])
        pred = pre.argmax(dim=-1)
        resmask = torch.gt(gVar(x[5]), 0)
        acc = (torch.eq(pred, gVar(x[5])) * resmask).float()#.mean(dim=-1)
        predres = (1 - acc) * pred.float() * resmask.float()
        accsum = torch.sum(acc, dim=-1)
        resTruelen = torch.sum(resmask, dim=-1).float()
        cnum = (torch.eq(accsum, resTruelen)).sum().float()
        if cnum.item() != 1:
            index += 1
            continue'''
        
        if index <= -1:
            index += 1
            continue
        print(dev_set.nl[index])
        ans = BeamSearch((x[0], x[1]), dev_set, model, 4, args.batch_size, index)
        index += 1
        for i in range(len(ans)):
            beam = ans[i]
            #print(beam[0].parent, beam[0].everTreepath, beam[0].state)
            f.write(beam.getTreestr())
            f.write("\n")
            f.write(str(beam.state) + "\n")
            f.flush()   
        #exit(0)
        #f.write(" ".join(ans.ans[1:-1]))
        #f.write("\n")
        #f.flush()#print(ans)
    #open("beams1.pkl", "wb").write(pickle.dumps(beamss))
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    if sys.argv[1] == "train": 
        train()
    elif sys.argv[1] == "eval": 
        eval()
    elif sys.argv[1] == "test": 
        test()
    else:
        testone()
     #test()




