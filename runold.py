import torch
from torch import optim
from Dataset import SumDataset,rs_collate_fn,Graph,ChunkedRandomSampler,rs_collate_fn1,pad_seq2
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
from stringfycode import stringfyNode
from transformers import AutoModel, AutoTokenizer
onelist = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers"]
#tokenizer = AutoTokenizer.from_pretrained('roberta-base')
from torch import multiprocessing as mp
import ssl
from torch.cuda import amp as torch_amp


import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
ssl._create_default_https_context = ssl._create_unverified_context
from tensorboardX import SummaryWriter
#sys.setrecursionlimit(500000000)
sys.setrecursionlimit(500000000)
#from pythonBottom.run import finetune
#from pythonBottom.run import pre
#wandb.init("sql")
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
args = dotdict({
    'NlLen':1024,
    'CodeLen':150,
    'batch_size':144,
    'TableLen':100,
    'embedding_size':512,
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
    "max_rel_pos":10,
    'par': True,
    'use_apex':False,
    'max_grad_norm':1.0,
    'use_torch_amp':False,
    "pretrain_name":"grammart5-small"
})
onelist = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers", 'annotation_argument_list', 'variable_declarator', 'throws', 'element_value_array_initializer', 'annotation_argument_list', 'switch_block_statement_group', 'class_body', 'catch_type', 'assert_statement', 'try_statement', 'local_variable_declaration', 'try_statement', 'constructor_body', 'type_parameters', 'resource_specification', 'inferred_parameters', 'try_with_resources_statement', 'inits', 'updates', 'conditions']
identifiers = ['identifier', 'type_identifier', 'null_literal', 'decimal_integer_literal', 'character_literal', 'decimal_floating_point_literal', 'hex_integer_literal', 'string_literal']
#os.environ["CUDA_VISIBLE_DEVICES"]="3, 0, 1, 2, 4, 5, 6, 7"
#os.environ['CUDA_LAUNCH_BLOCKING']="1"
def save_model(model, dirs='checkpointSearch/', optimizer=None, amp=None):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    if optimizer is not None:
        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'amp':amp.state_dict()
        }
        torch.save(checkpoint, dirs + 'best_model.ckpt')
    else:
        torch.save(model.state_dict(), dirs + 'best_model.ckpt')


def load_model(model, dirs = 'checkpointSearch/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt', map_location='cpu'))
use_cuda = True#torch.cuda.is_available()
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
def display(tab=None):
  experiment = comet_ml.get_global_experiment()
  experiment.display(tab=tab)
from Grape import Grape
import wandb
def splitbyCard(tmp, r, c, idx, cardnum=7):
    assert args.batch_size % cardnum == 0
    newbatchsize = args.batch_size // cardnum
    ans = []
    for i in range(7):
        tmpidx = tmp[0][idx[i][0]:idx[i][1], :]
        print(tmpidx)
        tmpidx[:, 0] = tmpidx[:,0] - i * newbatchsize
        tmpv = tmp[1][idx[i][0]:idx[i][1]]
        print(tmpidx.shape, tmpv.shape)
        print(tmpidx)
        tmpad = torch.sparse_coo_tensor(tmpidx.t(), tmpv, torch.Size([newbatchsize, r, c]))
        ans.append(tmpad)
    return ans
def train():
    #g = Grape(args, 'pythonrule.pkl')
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1)
    argc = parser.parse_args()
    local_rank = argc.local_rank

    torch.cuda.set_device('cuda:' + local_rank)
    dist.init_process_group(backend='nccl') 
    device = torch.device("cuda", int(local_rank))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_set = SumDataset(args, None, "train")
    args.cnum = len(train_set.ruledict)
    #print(len(train_set.edgedict))
    #print(args.bertnum)
    if dist.get_rank() == 0:
        wandb.init(project="code-gen")
        dev_set = SumDataset(args, None, "test")

    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.Vocsize = len(train_set.Char_Voc)
    args.rulenum = len(train_set.ruledict)
    #dev_set = SumDataset(args, "val")
    args.edgelen = len(train_set.edgedict)
    model = Decoder(args)
    model.encoder.model.resize_token_embeddings(args.rulenum)
    #load_model(model, 'checkModelNUM-304/')
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
        model = model.to(device)
        torch.cuda.manual_seed_all(args.seed)
        model = DDP(model, device_ids=[int(local_rank)], output_device='cuda:' + local_rank, find_unused_parameters=True)
        #from blcDP import BalancedDataParallel
        #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6])
        #BalancedDataParallel(0, model)#nn.DataParallel(model, device_ids=[0, 1])
    #model.to()
    global_step = 0
    model_filename = 'model.onnx'
    hasgraph = False
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_size = args.batch_size // 8
    for epoch in range(100000):
        j = 0
        train_sampler.set_epoch(epoch)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=4,collate_fn=rs_collate_fn, sampler=train_sampler, pin_memory=True)
        for dBatch in tqdm(data_loader):
            isBetter = False
            if j % 400 == 0 and dist.get_rank() == 0:
                devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=True, num_workers=4,collate_fn=rs_collate_fn)
                model = model.eval()
                accs = []
                tcard = []
                tmp = []
                for devBatch in tqdm(devloader):
                    for i in range(len(devBatch)):
                        if i == 3:
                            tmpad = torch.sparse_coo_tensor(devBatch[i][0].t(), devBatch[i][1], torch.Size([devBatch[0].size(0), devBatch[1].size(1), devBatch[1].size(1)])).to_dense()
                        else:
                            devBatch[i] = gVar(devBatch[i])
                    antimask = gVar(getAntiMask(devBatch[1].size(1)))
                    antimask2 = antimask.unsqueeze(0).repeat(args.batch_size, 1, 1).unsqueeze(1)

                    with torch.no_grad():
                        bsize = devBatch[0].size(0)
                        antimask2 = antimask2[:bsize]
                        _, pre = model(devBatch[0], devBatch[1], devBatch[4], antimask2, devBatch[2])
                        if False:
                            torch.onnx.export(model.module, (devBatch[0], devBatch[1], devBatch[4], antimask2, devBatch[2]), model_filename, opset_version=11)
                            writer.add_onnx_graph(model_filename)
                            hasgraph = True
                            display("assets")
                        #print(devBatch[2].size())
                        pred = pre.argmax(dim=-1)
                        resmask = torch.ne(devBatch[2], args.mask_id)
                        acc = (torch.eq(pred, devBatch[2]) * resmask).float()#.mean(dim=-1)
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
                wandb.log({"acc": acc, "tnum": tnum})
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
            model = model.train()
            for i in range(len(dBatch)):
                if i == 3:
                    spad = torch.sparse_coo_tensor(dBatch[i][0].t(), dBatch[i][1], torch.Size([dBatch[0].size(0), dBatch[1].size(1), dBatch[1].size(1)])).to_dense()
                else:
                    dBatch[i] = gVar(dBatch[i])
            antimask = gVar(getAntiMask(dBatch[1].size(1)))
            antimask2 = antimask.unsqueeze(0).repeat(train_size, 1, 1).unsqueeze(1)
            loss, _ = model(dBatch[0], dBatch[1], dBatch[4], antimask2, dBatch[2])
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
            resmask = torch.ne(dBatch[2], args.mask_id)
            #print(torch.sum(resmask), torch.sum(loss))

            loss = torch.sum(loss) / torch.sum(resmask)
            if loss.sum().item() == np.inf:
                print('inf')
                exit(0)
            if loss.item() == np.inf:
                print(j)
                assert(0)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()
            if j % args.gradient_accumulation_steps == 0:
                optimizer.step()#_and_update_lr()
                optimizer.zero_grad()
            if dist.get_rank() == 0:
                wandb.log({"loss": loss.item()})
            #wandb.log({"loss": loss.item()})
            j += 1
            global_step += 1
            #display("metrics")
            #display("assets")
def pretrain():
    if args.par:
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", default=-1)
        argc = parser.parse_args()
        local_rank = argc.local_rank
        torch.cuda.set_device('cuda:' + local_rank)
        dist.init_process_group(backend='nccl') 
        if dist.get_rank() == 0:
            wandb.init(project="grammar-t5")
    torch.manual_seed(args.seed)        
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.par:
        train_set = SumDataset(args, None, "train", idx=int(dist.get_rank()))        
        device = torch.device("cuda", int(local_rank))
    else:
        train_set = SumDataset(args, None, "train", idx=0)
        device = torch.device("cuda", 0)

    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.rulenum = train_set.rulenum
    model = Decoder(args)
    model.encoder.model.resize_token_embeddings(args.rulenum)
    base_params = list(map(id, model.encoder.model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": 2e-4, 'notchange':False},
        {"params": model.encoder.model.parameters(), "lr": 5e-5, 'notchange':True},
    ]
    optimizer = optim.AdamW(params, eps=1e-8)
    scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=100000, init_lr0=5e-5, init_lr2=2e-4, end_learning_rate=0.000, power=1.0)
    pathnames = []
    num_trial = patience = 0
    isBetter = False
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, max_steps=50000)
    maxAcc= 0
    maxC = 0
    minloss = 1e10
    if use_cuda:
        model = model.cuda()
        print('using GPU')
        if args.use_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        torch.cuda.manual_seed_all(args.seed)
        if args.par:
            model = DDP(model, device_ids=[int(local_rank)], output_device='cuda:' + local_rank, find_unused_parameters=True)
        else:
            model = model#nn.DataParallel(model)
        #from blcDP import BalancedDataParallel
        #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6])
        #BalancedDataParallel(0, model)#nn.DataParallel(model, device_ids=[0, 1])
    #model.to()
    global_step = 0
    train_size = args.batch_size // 8
    if args.use_torch_amp:
        scaler = torch_amp.GradScaler()
    for epoch in range(100000):
        j = 0    
        sampler = ChunkedRandomSampler(train_set, train_size)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=10,collate_fn=rs_collate_fn, sampler=sampler, pin_memory=True)
        for dBatch in tqdm(data_loader):
            if j % 10000 == 1000 and dist.get_rank() == 0:
                if len(pathnames) < 10:
                    save_model(model.module, 'checkpointEpchLR%dIter%d/'%(epoch, j))#print(loss.item())
                    pathnames.append('checkpointEpchLR%dIter%d/'%(epoch, j))
                else:
                    os.system('rm -r %s' % pathnames[0])
                    pathnames.pop(0)
                    save_model(model.module, 'checkpointEpchLR%dIter%d/'%(epoch, j))#print(loss.item())
                    pathnames.append('checkpointEpchLR%dIter%d/'%(epoch, j))
            model = model.train()
            for x in dBatch:
                for k in dBatch[x]:
                    dBatch[x][k] = gVar(dBatch[x][k])
                #dBatch[x] = gVar(dBatch[x])
            tdBatch = dBatch
            #iden
            dBatch = tdBatch['iden'] 
            antimask = gVar(getAntiMask(dBatch['res'].size(1) - 1))
            antimask2 = antimask.unsqueeze(0).repeat(train_size, 1, 1).unsqueeze(1)
            if args.use_torch_amp:
                with torch_amp.autocast():
                    loss, _ = model(dBatch['nl'], dBatch['res'], dBatch['parent'], antimask2, lefttree=True)
            else:
                loss, _ = model(dBatch['nl'], dBatch['res'], dBatch['parent'], antimask2, lefttree=True)

            resmask = torch.ne(dBatch['res'][:,1:], args.mask_id)
            loss = torch.sum(loss) / torch.sum(resmask)

            dBatch = tdBatch['rule']
            antimask = gVar(getAntiMask(dBatch['res'].size(1) - 1))
            antimask2 = antimask.unsqueeze(0).repeat(train_size, 1, 1).unsqueeze(1)
            if args.use_torch_amp:
                with torch_amp.autocast():
                    loss2, _ = model(dBatch['nl'], dBatch['res'], dBatch['parent'], antimask2, lefttree=True)
            loss2, _ = model(dBatch['nl'], dBatch['res'], dBatch['parent'], antimask2, lefttree=True)
            resmask = torch.ne(dBatch['res'][:,1:], args.mask_id)
            loss2 = torch.sum(loss2) / torch.sum(resmask)
            loss3 = 0.5 * loss + 0.5 * loss2
            if loss.item() == np.inf:
                print(j)
                assert(0)
            if args.gradient_accumulation_steps > 1:
                loss = loss3 / args.gradient_accumulation_steps
            if args.use_apex:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler_poly_lr_decay.step()
            elif args.use_torch_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler_poly_lr_decay.step()
            else:
                loss.backward()
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler_poly_lr_decay.step()
            if args.par and dist.get_rank() == 0:
                wandb.log({"loss": loss.item() * 10})
            #wandb.log({"loss": loss.item()})
            j += 1
            global_step += 1
            #display("metrics")
            #display("assets")
def pretrain2():
    if args.par:
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", default=-1)
        argc = parser.parse_args()
        local_rank = argc.local_rank
        torch.cuda.set_device('cuda:' + local_rank)
        dist.init_process_group(backend='nccl') 
        if dist.get_rank() == 0:
            wandb.init(project="grammar-t5")
    torch.manual_seed(args.seed)        
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.par:
        train_set = SumDataset(args, None, "train", idx=int(dist.get_rank()), mode='gen')        
        device = torch.device("cuda", int(local_rank))
    else:
        train_set = SumDataset(args, None, "train", idx=0, mode='gen')
        device = torch.device("cuda", 0)

    args.Nl_Vocsize = len(train_set.Nl_Voc)
    args.rulenum = train_set.rulenum
    model = Decoder(args)
    model.encoder.model.resize_token_embeddings(args.rulenum)
    base_params = list(map(id, model.encoder.model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": 2e-4, 'notchange':False},
        {"params": model.encoder.model.parameters(), "lr": 5e-5, 'notchange':True},
    ]
    optimizer = optim.AdamW(params, eps=1e-8)
    scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=100000, init_lr0=5e-5, init_lr2=2e-4, end_learning_rate=0.000, power=1.0)
    pathnames = []
    num_trial = patience = 0
    isBetter = False
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, max_steps=50000)
    maxAcc= 0
    maxC = 0
    minloss = 1e10
    if use_cuda:
        model = model.cuda()
        print('using GPU')
        if args.use_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        torch.cuda.manual_seed_all(args.seed)
        load_model(model, 'checkpointEpchLR8Iter11000/')
        if args.par:
            model = apex.parallel.DistributedDataParallel(model)#DDP(model, device_ids=[int(local_rank)], output_device='cuda:' + local_rank, find_unused_parameters=True)
        else:
            model = model#nn.DataParallel(model)
        #from blcDP import BalancedDataParallel
        #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6])
        #BalancedDataParallel(0, model)#nn.DataParallel(model, device_ids=[0, 1])
    #model.to()
    global_step = 0
    train_size = args.batch_size // 8
    if args.use_torch_amp:
        scaler = torch_amp.GradScaler()
    for epoch in range(100000):
        j = 0    
        sampler = ChunkedRandomSampler(train_set, train_size)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=10,collate_fn=rs_collate_fn1, sampler=sampler, pin_memory=True)
        for dBatch in tqdm(data_loader):
            if j % 10000 == 1000 and dist.get_rank() == 0:
                if len(pathnames) < 10:
                    save_model(model.module, 'checkpointEpch%dIter%d/'%(epoch, j))#print(loss.item())
                    pathnames.append('checkpointEpch%dIter%d/'%(epoch, j))
                else:
                    os.system('rm -r %s' % pathnames[0])
                    pathnames.pop(0)
                    save_model(model.module, 'checkpointEpch%dIter%d/'%(epoch, j))#print(loss.item())
                    pathnames.append('checkpointEpch%dIter%d/'%(epoch, j))
            model = model.train()
            for x in dBatch:
                dBatch[x] = gVar(dBatch[x])
                #dBatch[x] = gVar(dBatch[x]) 
            antimask = gVar(getAntiMask(dBatch['res'].size(1) - 1))
            antimask2 = antimask.unsqueeze(0).repeat(train_size, 1, 1).unsqueeze(1)
            if args.use_torch_amp:
                with torch_amp.autocast():
                    loss, _ = model(dBatch['nl'], dBatch['res'], dBatch['parent'], antimask2, lefttree=False)
            else:
                loss, _ = model(dBatch['nl'], dBatch['res'], dBatch['parent'], antimask2, lefttree=False)

            resmask = torch.ne(dBatch['res'][:,1:], args.mask_id)
            loss = torch.sum(loss) / torch.sum(resmask)
            loss3 = loss
            if loss.item() == np.inf:
                print(j)
                assert(0)
            if args.gradient_accumulation_steps > 1:
                loss = loss3 / args.gradient_accumulation_steps
            if args.use_apex:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler_poly_lr_decay.step()
            elif args.use_torch_amp:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if j % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler_poly_lr_decay.step()
            else:
                loss.backward()
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler_poly_lr_decay.step()
            if args.par and dist.get_rank() == 0:
                wandb.log({"loss": loss.item() * 10})
            #wandb.log({"loss": loss.item()})
            j += 1
            global_step += 1
            #display("metrics")
            #display("assets")
def mergeIdentifier(root):
    if root.name in identifiers:
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
        self.id = -1

def getiden(root):
    if len(root.child) == 0:
        return root.name[:-4]
    ans = ""
    for x in root.child:
        ans += getiden(x)
    return ans

class SearchNode:
    def __init__(self, ds, nl, name='', mode='gen', idenids=None):
        if mode == 'iden':
            self.currIdenIdx = 0
            self.idenids = idenids
            self.state = [ds.ruledict['<extra_id_0>']]
            self.newidens = []
        else:
            self.state = [ds.ruledict["start -> java"]]
        self.prob = 0
        self.aprob = 0
        self.bprob = 0
        self.finish = False
        if mode == 'iden':
            self.root = Node(name, 0)
        else:
            self.root = Node("java", 2)
        self.parent = [0]
        #self.parent[args.NlLen]
        self.expanded = None
        #self.ruledict = ds.rrdict
        self.expandedname = []
        self.child = {}
        for x in ds.ruledict:
            self.expandedname.append(x.strip().split()[0])
        self.expandedname.extend(onelist)
    def selcetNode(self, root):
        if not root.expanded and (root.name in self.expandedname) and root.name not in onelist and root.name not in identifiers:
            return root
        else:
            for x in root.child:
                ans = self.selcetNode(x)
                if ans:
                    return ans
            if (root.name in onelist or root.name in identifiers) and root.expanded == False:
                return root
        return None
    def selectExpandedNode(self):
        self.expanded = self.selcetNode(self.root)
    def getRuleEmbedding(self, ds):      
        inputrule = ds.pad_seq(self.state, ds.Code_Len)
        return inputrule
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
                if self.expanded.name not in identifiers:
                    return False
        else:
            rules = ds.rrdict[rule]
            if rules.strip().split()[0].lower() != self.expanded.name.lower():
                return False
        return True
    def applyrule(self, rule, ds, mode='gen'):
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
        self.expanded.id = len(self.state)
        if mode == 'iden':
            if 'extra_id' in rules:
                self.parent.append(0)
            else:
                self.parent.append(self.idenids[self.currIdenIdx])
        else:
            self.parent.append(self.expanded.fatherlistID)
        #self.graph.addEdge(len(self.state), self.expanded.fatherlistID, 1)
        #if self.expanded.father:
        #    if len(self.expanded.child) >= 2:
        #        self.graph.addEdge(len(self.state), self.expanded.child[-2].id, 1)
        assert(self.expanded.fatherlistID != -1)
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
        else:
            self.state.append(rule)
        if self.expanded.name not in onelist:
            self.expanded.expanded = True
        if self.expanded.name in identifiers: #self.expanded.name in ['qualifier', 'member', 'name', 'value', 'flag']:
            if 'Ġ' in rules:
                self.expanded.child.reverse()
                self.expanded.expanded = True
                if self.root.name == 'identifier':
                    self.newidens.append(getiden(self.root))
                    self.currIdenIdx += 1
                    if self.currIdenIdx < len(self.idenids):
                        self.root = Node('identifier', 0)
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


def BeamSearch(inputnl, vds, model, beamsize, batch_size, k, queue=None, name='', mode='gen'):
    #print(inputnl[0].shape)
    batch_size = gVar(inputnl[0]).size(0)
    args.NlLen = inputnl[0].shape[1]
    #print('------------------1')
    #print(inputnl[3][0])
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            if mode == 'iden':
                beams[i] = [SearchNode(vds, [], name=name, mode=mode, idenids=inputnl[1])]
            else:
                beams[i] = [SearchNode(vds, [])]
        nlencode, nlmask = model.nl_encode(gVar(torch.tensor(inputnl[0])))
        index = 0
        endnum = {}
        continueSet = {}
        codelen = 250
        showtqdm = tqdm(range(codelen))
        while True:
            #print(index)
            showtqdm.update(1)
            args.CodeLen = min(index + 2, codelen)
            vds.Code_Len = min(index + 2, codelen)#index + 1
            tmpbeam = {}
            ansV = {}
            if len(endnum) == batch_size:
                break
            #if index > 10:
            #    assert(0)
            if index >= codelen:
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
                    #print(x.printTree(x.root))
                    if not x.finish:
                        x.selectExpandedNode()
                    if x.expanded == None or len(x.state) >= args.CodeLen:
                        x.finish = True
                        ansV.setdefault(i, []).append(x)
                    else:
                        validnum.append(i)
                        a = x.getRuleEmbedding(vds)
                        tmprule.append(a)
                        #tmpAd.append(inputnl[3][0][:args.CodeLen, :args.CodeLen])
                        
                        tmpAd.append(pad_seq2(x.parent, args.CodeLen))
                #print("--------------------------")
                if len(tmprule) == 0:
                    continue
                batch_sizess = len(tmprule)
                antimask2 = antimask.unsqueeze(0).repeat(batch_sizess, 1, 1).unsqueeze(1)
                tmprule = np.array(tmprule)
                tmpAd = np.array(tmpAd)
                bsize = batch_sizess
                antimask2 = antimask2[:bsize]
                #print('------------------2')
                #print(tmpAd[0])
                if mode == 'iden': 
                    result = model.test_foward(nlencode[validnum], nlmask[validnum], gVar(tmprule))
                else:
                    result = model.test_foward(nlencode[validnum], nlmask[validnum], gVar(tmprule))
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
                        if tmpbeamsize >= 20 or i > 150:
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
                            copynode.applyrule(x[1], vds, mode)
                            copynode.prob = x[0]
                            beams[j].append(copynode)
                        else:
                            beams[j].append(x[2])
            index += 1
            
        return beams
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
    s = "<nl> return the sum of a and b"
    dev_set.processOne(s)
    load_model(model, "checkModelNUM/")
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
    dev_set = SumDataset(args, None, "test")
    args.cnum = len(dev_set.ruledict)
    args.Nl_Vocsize = len(dev_set.Nl_Voc)
    args.Vocsize = len(dev_set.Char_Voc)
    args.rulenum = len(dev_set.ruledict)
    #print(len(train_set.edgedict))
    print(args.bertnum)
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    model = Decoder(args)

    load_model(model, 'checkModel/')
    if use_cuda:#torch.cuda.is_available():
        print('using GPU')
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=[0, 1])
    
    model = model.module
    ctx = mp.get_context('spawn')
    args.batch_size = 250
    print(len(dev_set))
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0, collate_fn=rs_collate_fn)
    model = model.eval()
    #load_model(model)
    f = open("outval1.txt", "w")
    index = 0 
    for x in tqdm(devloader):
        #if index <= 0:
        #    index += 1
        #    continue
        #print(dev_set.nl[index])
        ans = BeamSearch((x[0], x[1], x[2], x[3]), dev_set, model, 4, args.batch_size, index)
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
def testnl():
    #pre()
    args = json.load(open("config.json", "r"))
    args = dotdict(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev_set = SumDataset(args, None, "testone")

    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    #print(dev_set.Nl_Voc)
    model = Decoder(args)
    model.encoder.model.resize_token_embeddings(args.rulenum)
    s = '''<nl> returns a hash of the given files contents . reads the file fully into memory before hashing so only use with small files . concode_field_sep Sha256Hash ZERO_HASH concode_elem_sep byte[] bytes concode_field_sep Sha256Hash createDouble concode_elem_sep int hashCode concode_elem_sep boolean equals concode_elem_sep Sha256Hash create concode_elem_sep BigInteger toBigInteger concode_elem_sep String toString concode_elem_sep Sha256Hash duplicate concode_elem_sep int compareTo concode_elem_sep byte[] getBytes'''
    load_model(model, "checkModelNUM/")
    if use_cuda:
        print('using GPU')
        model = model.cuda()    
    from transformers import AutoTokenizer  
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    '''train_set = SumDataset(args, None, "test", idx=0)
    devdataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, drop_last=True, num_workers=10,collate_fn=rs_collate_fn1, shuffle=False, pin_memory=True)    


    for l, dBatch in enumerate(devdataloader):
        for key in dBatch:
            dBatch[key] = gVar(dBatch[key])    
        antimask = gVar(getAntiMask(dBatch['res'].size(1) -1))
        antimask2 = antimask.unsqueeze(0).repeat(1, 1, 1).unsqueeze(1)
        print(dBatch['parent'])
        loss, _ = model(dBatch['nl'], dBatch['res'], dBatch['parent'], antimask2, lefttree=False)
        nl = tokenizer.convert_ids_to_tokens(dBatch['nl'][0].cpu().numpy())
        print(" ".join(nl).replace("Ġ", "")) 
        resmask = torch.ne(dBatch['res'], args.mask_id)
        print(loss.sum() / resmask.sum())'''
        



    args.batch_size = 1
    model = model.eval()
    f = open("outval1.txt", "w")
    index = 0 

    ans = BeamSearch((torch.tensor([tokenizer.encode(s.replace('concode_elem_sep', tokenizer.sep_token).replace('concode_field_sep', tokenizer.sep_token))]), []), dev_set, model, 5, args.batch_size, index, name="", mode='gen')
    for i in range(len(ans)):
        beam = ans[i]
        for candidates in ans[i]:
            print(stringfy(candidates.root))
        #f.write(beam.getTreestr())
        #f.write("\n")
        #f.write(str(beam.state) + "\n")
def finetune():
    if args.par:
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", default=-1)
        argc = parser.parse_args()
        local_rank = argc.local_rank
        train_set = SumDataset(args, None, "train", mode='finetune', idx=int(local_rank))

        torch.cuda.set_device('cuda:' + local_rank)
        dist.init_process_group(backend='nccl') 
        device = torch.device("cuda", int(local_rank))
    else:
        #device = torch.device("cuda:0")
        train_set = SumDataset(args, None, "train", mode='finetune', idx=3)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.cnum = len(train_set.ruledict)
    args.rulenum = train_set.rulenum
    #print(len(train_set.edgedict))
    #print(args.rulenum)
    if args.par and dist.get_rank() == 0:
        wandb.init(project="code-gen")
        dev_set = SumDataset(args, None, "test")

    model = Decoder(args)
    model.encoder.model.resize_token_embeddings(args.rulenum)
    load_model(model, "checkpointEpch60Iter11000/")
    base_params = list(map(id, model.encoder.model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": 5e-5, 'notchange':False},
        {"params": model.encoder.model.parameters(), "lr": 5e-5, 'notchange':False},
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
        if args.par:
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        #DDP(model, device_ids=[int(local_rank)], output_device='cuda:' + local_rank, find_unused_parameters=True)
        #from blcDP import BalancedDataParallel
        #model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6])
        #BalancedDataParallel(0, model)#nn.DataParallel(model, device_ids=[0, 1])
    #model.to()
    global_step = 0
    hasgraph = False
    train_size = args.batch_size // 8
    for epoch in range(100000):
        j = 0
        sampler = ChunkedRandomSampler(train_set, train_size)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=4,collate_fn=rs_collate_fn1, sampler=sampler, pin_memory=True)
        for dBatch in tqdm(data_loader):
            isBetter = False
            if j % 400 == 300 and dist.get_rank() == 0:
                devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=20,
                                              shuffle=False, drop_last=True, num_workers=4,collate_fn=rs_collate_fn1)
                model = model.eval()
                accs = []
                tcard = []
                tmp = []
                for devBatch in tqdm(devloader):
                    for x in devBatch:
                        devBatch[x] = gVar(devBatch[x])
                    antimask = gVar(getAntiMask(devBatch['res'].size(1) - 1))
                    antimask2 = antimask.unsqueeze(0).repeat(20, 1, 1).unsqueeze(1)

                    with torch.no_grad():
                        bsize = devBatch['nl'].size(0)
                        antimask2 = antimask2[:bsize]
                        _, pre = model(devBatch['nl'], devBatch['res'], devBatch['parent'], antimask2, lefttree=False)
                        pred = pre.argmax(dim=-1)
                        resmask = torch.ne(devBatch['res'][:,1:], args.mask_id)
                        acc = (torch.eq(pred, devBatch['res'][:,1:]) * resmask).float()#.mean(dim=-1)
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
                wandb.log({"acc": acc, "tnum": tnum})
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
            model = model.train()
            for x in ((dBatch)):
                dBatch[x] = gVar(dBatch[x])
            antimask = gVar(getAntiMask(dBatch['res'].size(1) - 1))
            antimask2 = antimask.unsqueeze(0).repeat(train_size, 1, 1).unsqueeze(1)
            loss, _ = model(dBatch['nl'], dBatch['res'], dBatch['parent'], antimask2, lefttree=False)
            #print(dBatch['nl'].size(), dBatch['res'].size(), dBatch['parent'].size(), antimask2.size())
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
            resmask = torch.ne(dBatch['res'][1:], args.mask_id)
            #print(torch.sum(resmask), torch.sum(loss))

            loss = torch.sum(loss) / torch.sum(resmask)
            if loss.sum().item() == np.inf:
                print('inf')
                exit(0)
            if loss.item() == np.inf:
                print(j)
                assert(0)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()
            if j % args.gradient_accumulation_steps == 0:
                optimizer.step()#_and_update_lr()
                optimizer.zero_grad()
            if args.par and dist.get_rank() == 0:
                wandb.log({"loss": loss.item()})
            #wandb.log({"loss": loss.item()})
            j += 1
            global_step += 1
            #display("metrics")
            #display("assets")
import line_profiler
lstgr = [0, 1, 2, 3, 4, 7, 8, 9, 11, 13, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 101, 102, 103, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 120, 121, 122, 123, 125, 126, 127, 128, 131, 132, 133, 134, 135, 137, 139, 141, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212, 214, 215, 217, 218, 219, 220, 223, 225, 226, 227, 230, 231, 232, 233, 236, 238, 239, 240, 243, 245, 246, 247, 250, 256, 257, 259, 260, 261, 262, 264, 265, 266, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 289, 290, 291, 292, 293, 294, 297, 298, 299, 300, 302, 303, 304, 306, 307, 308, 309, 312, 313, 314, 315, 317, 318, 323, 325, 327, 329, 331, 332, 334, 335, 336, 338, 339, 340, 342, 344, 345, 346, 347, 348, 349, 350, 351, 354, 355, 357, 359, 361, 365, 367, 368, 369, 370, 371, 373, 374, 375, 376, 378, 379, 384, 385, 387, 388, 389, 390, 391, 392, 394, 395, 396, 397, 398, 400, 401, 402, 403, 404, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 421, 423, 424, 425, 426, 428, 429, 430, 431, 432, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 479, 480, 481, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 513, 514, 515, 516, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 542, 543, 544, 545, 548, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 562, 565, 567, 568, 570, 571, 572, 573, 576, 577, 578, 580, 582, 583, 584, 587, 588, 589, 590, 591, 593, 594, 595, 596, 597, 598, 599, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 612, 613, 614, 620, 622, 623, 624, 625, 627, 629, 630, 631, 632, 633, 634, 638, 639, 640, 641, 642, 643, 645, 646, 647, 648, 649, 651, 652, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 690, 691, 693, 695, 696, 697, 698, 700, 702, 703, 704, 706, 707, 708, 709, 711, 712, 713, 714, 715, 716, 717, 718, 720, 721, 722, 723, 724, 725, 726, 727, 729, 730, 732, 734, 735, 738, 739, 740, 741, 742, 743, 745, 746, 747, 749, 750, 753, 754, 757, 758, 759, 760, 761, 762, 763, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 776, 777, 778, 779, 781, 783, 784, 786, 787, 789, 790, 792, 795, 796, 797, 799, 800, 801, 802, 804, 805, 806, 807, 808, 809, 810, 812, 813, 815, 816, 817, 818, 819, 820, 821, 822, 824, 826, 827, 828, 829, 830, 831, 832, 834, 835, 839, 840, 843, 844, 845, 846, 847, 849, 850, 852, 853, 858, 859, 860, 861, 862, 863, 864, 865, 867, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 910, 911, 913, 914, 915, 916, 917, 918, 920, 921, 922, 923, 924, 925, 926, 928, 930, 931, 932, 934, 935, 936, 937, 938, 939, 941, 942, 944, 945, 947, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 976, 977, 978, 979, 981, 982, 983, 984, 986, 987, 989, 991, 992, 994, 995, 996, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1013, 1015, 1016, 1017, 1018, 1019, 1020, 1022, 1023, 1024, 1025, 1026, 1028, 1029, 1030, 1031, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1044, 1045, 1046, 1047, 1048, 1049, 1051, 1052, 1053, 1054, 1056, 1057, 1058, 1059, 1060, 1062, 1063, 1064, 1065, 1066, 1067, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1087, 1088, 1090, 1092, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1145, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1160, 1161, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1173, 1175, 1177, 1178, 1179, 1180, 1181, 1182, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1210, 1212, 1213, 1214, 1216, 1217, 1218, 1219, 1220, 1222, 1223, 1225, 1229, 1230, 1231, 1233, 1235, 1236, 1238, 1240, 1241, 1242, 1244, 1245, 1246, 1247, 1248, 1250, 1253, 1254, 1255, 1256, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1314, 1315, 1316, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1345, 1346, 1348, 1350, 1351, 1352, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1366, 1369, 1370, 1371, 1372, 1375, 1376, 1377, 1378, 1379, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1477, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1495, 1497, 1499, 1501, 1502, 1504, 1506, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1520, 1521, 1523, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1534, 1535, 1536, 1537, 1538, 1539, 1541, 1543, 1545, 1546, 1547, 1548, 1550, 1551, 1552, 1553, 1554, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1570, 1571, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1586, 1587, 1588, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1598, 1600, 1602, 1603, 1604, 1606, 1607, 1608, 1610, 1611, 1612, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1625, 1626, 1627, 1628, 1629, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1681, 1682, 1683, 1684, 1685, 1687, 1688, 1690, 1691, 1693, 1694, 1695, 1697, 1698, 1699, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1710, 1711, 1712, 1713, 1714, 1715, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1725, 1726, 1727, 1729, 1731, 1732, 1733, 1735, 1736, 1737, 1738, 1739, 1741, 1743, 1744, 1745, 1746, 1748, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1762, 1766, 1767, 1768, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798]

@torch.no_grad()
def evalmodelacc(dev_set, model, device):
    global args
    data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=25, drop_last=False, num_workers=2,collate_fn=rs_collate_fn1, shuffle=False, pin_memory=True)
    accs = []
    tnums = []
    model = model.eval()
    lst = []
    for i, dbatch in enumerate(data_loader):
        dbatch['nl'] = dbatch['nl'].to(device)
        dbatch['res'] = dbatch['res'].to(device)
        loss, pred = model(dbatch['nl'], dbatch['res'])
        pre = pred.argmax(dim=-1)
        resmask = torch.ne(dbatch['res'][:,1:], args.mask_id)
        accnum = (torch.eq(dbatch['res'][:,1:], pre) * resmask).sum(dim=-1)
        acc = accnum.float() / resmask.sum(dim=-1).float()
        accs.append(acc.mean().item())
        tnum = torch.eq(accnum, resmask.sum(dim=-1))
        for j in range(len(tnum)):
            if tnum[j].item() == 1:
                lst.append(i * 25 + j)
                if i * 25 + j + 300 * args.local_rank not in lstgr:
                    print(i * 25 + j + + 300 * args.local_rank)
        tnums.append(tnum.sum().item())
    tnum = np.sum(tnums)
    acc = np.mean(accs)
    return tnum, acc, lst


def testbatch():
    global args
    task = sys.argv[2]
    taskconfig = json.loads(open("processdata/data/%s/config.json" % task).read())
    lang = taskconfig['lang']
    if task in ['commentjava', 'commentpython']:
        mode = 'nl'
    else:
        mode = 'gen'
    local_rank = sys.argv[1]
    args.pretrain_name = "grammart5-base"
    #pre()
    #args = json.load(open("config.json", "r"))
    #args = dotdict(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.local_rank = int(local_rank)
    args.NlLen = taskconfig['NlLen']
    args.CodeLen = taskconfig['CodeLen']

    dev_set = SumDataset(args, None, "test", idx=int(local_rank))
    rdic = {}
    for x in dev_set.Nl_Voc:
        rdic[dev_set.Nl_Voc[x]] = x
    args.rulenum = dev_set.rulenum
    #print(dev_set.Nl_Voc)
    model = Decoder(args)
    newruledic = pickle.load(open("processdata/%srules.pkl"%task, "rb"))
    print(len(newruledic))
    model.resize_token_embeddings(len(newruledic))
    #load_model(model, "finetune-models-base/checkModel%s/"%task)
    load_model(model, "checkModel%s/"%task)
    if use_cuda:
        print('using GPU')
        model = model.cuda() 
    #tnum, belu, res = evalmodelacc(dev_set, model, 'cuda:0')
    #print(tnum, belu)
    #res = [x + 300 * int(local_rank) for x in res]
    #print(res)
    args.batch_size = 10
    model = model.eval()
    f = open("outval%d.txt"%int(local_rank), "w")
    index2 = 0
    #print(len(dev_set))
    dataloader = torch.utils.data.DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=rs_collate_fn1, pin_memory=True) 
    from beamsearch import BeamSearch
    beamsize = taskconfig['beamsize']
    lp = taskconfig['length_penalty']
    beam = BeamSearch(beamsize, newruledic, lp)
    for j, dBatch in enumerate(tqdm(dataloader)):
        #if j != 59:
        #    continue
        #print(dBatch['res'])
        dBatch['nl'] = dBatch['nl'].cuda().repeat_interleave(beamsize, dim=0)
        ans = beam.search(dBatch['nl'], model, lang=lang, max_len=args.CodeLen, vocabsize=len(newruledic), mode=mode)        
        for i in range(len(ans)):
            #for beamone in ans[i]:
                beamone = ans[i].set[0]
                #for candidates in ans[i]:
                #    print(stringfy(candidates.root))
                root = beam.convertrulelist2tree(beamone.state, lang=lang, mode=mode)
                if isinstance(root, str):
                    f.write(root + '\n')
                else:
                    f.write(root.printTree(root))
                    f.write("\n")
                    f.write(str(beamone.state) + "\n")
                f.flush()
        index2 += 1
    print(local_rank)
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    testbatch()    
    '''if sys.argv[1] == "train": 
        train()
    elif sys.argv[1] == "eval": 
        eval()
    elif sys.argv[1] == "test": 
        profile = line_profiler.LineProfiler()
        profile.enable()
        test()
        profile.disable()
        profile.print_stats(sys.stdout)
    else:
        testone()'''
     #test()




