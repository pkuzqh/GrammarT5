import torch
from torch import optim
from Dataset import SumDataset,rs_collate_fn,Graph,ChunkedRandomSampler,rs_collate_fn1
import os
import time
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
from scheduler import PolynomialLRDecay
from transformers import AutoModel, AutoTokenizer
onelist = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers"]
#tokenizer = AutoTokenizer.from_pretrained('roberta-base')
from torch import multiprocessing as mp
from comet_ml import Experiment
import comet_ml
import ssl
from apex import amp
import apex
from torch.cuda import amp as torch_amp
from accelerate import Accelerator

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
    'NlLen':512,
    'CodeLen':512,
    'batch_size':144,
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
    "gradient_accumulation_steps":20,
    "patience":15,
    "max_num_trials":8,
    "max_rel_pos":10,
    'par': True,
    'use_apex':False,
    'max_grad_norm':1.0,
    'use_torch_amp':False,
    'mix_precision':'fp16',
    'task':'django',
    'eval':True,
    "pretrain_name":"grammart5-small"
})
onelist = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers", 'annotation_argument_list', 'variable_declarator', 'throws', 'element_value_array_initializer', 'annotation_argument_list', 'switch_block_statement_group', 'class_body', 'catch_type', 'assert_statement', 'try_statement', 'local_variable_declaration', 'try_statement', 'constructor_body', 'type_parameters', 'resource_specification', 'inferred_parameters', 'try_with_resources_statement']
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
from transformers import get_linear_schedule_with_warmup
def pretrain():
    args.batch_size = 20 * 6 * 20
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='bf16', log_with='wandb')
    hps = {"num_iterations": 100, "learning_rate": 2e-4}
    
    accelerator.init_trackers("grammar-t5", config=hps)
    torch.manual_seed(args.seed)        
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_set = SumDataset(args, None, "train", idx=accelerator.process_index)        
    device = accelerator.device
    totoalnumber = len(train_set) * accelerator.num_processes
    args.rulenum = train_set.rulenum
    model = Decoder(args)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=50 * totoalnumber // args.batch_size)
    #load_model(model, dirs = 'checkpointEpchLR5Iter10/')
    pathnames = []
    #data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=15,collate_fn=rs_collate_fn, sampler=sampler, pin_memory=True)
    global_step = 0
    train_size = args.batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_size
    #model, optimizer, _ = accelerator.prepare(model, optimizer, data_loader)
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(model)
    accelerator.register_for_checkpointing(optimizer)
    accelerator.register_for_checkpointing(scheduler)
    for epoch in range(100):_checkpointing(scheduler)
    for epoch in range(100):
        j = 0    
        sampler = ChunkedRandomSampler(train_set, train_size)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=15,collate_fn=rs_collate_fn, sampler=sampler, pin_memory=True)
        for dBatch in tqdm(data_loader):
            if j % 10000 == 10 and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                if len(pathnames) < 10:
                    save_model(unwrapped_model, 'checkpointEpchLR%dIter%d/'%(epoch, j))#print(loss.item())
                    #accelerator.save_state('checkpointEpchLR%dIter%d/'%(epoch, j))
                    pathnames.append('checkpointEpchLR%dIter%d/'%(epoch, j))
                else:
                    os.system('rm -r %s' % pathnames[0])
                    pathnames.pop(0)
                    save_model(unwrapped_model, 'checkpointEpchLR%dIter%d/'%(epoch, j))#print(loss.item())
                    #accelerator.save_state('checkpointEpchLR%dIter%d/'%(epoch, j))
                    pathnames.append('checkpointEpchLR%dIter%d/'%(epoch, j))
            accelerator.wait_for_everyone()
            model.train()
            for x in dBatch:
                dBatch[x] = dBatch[x].to(device)
                #for k in dBatch[x]:
                #    dBatch[x][k] = dBatch[x][k].to(device)
            tdBatch = dBatch
            #iden
            #dBatch = tdBatch['iden'] 
            loss, _ = model(dBatch['nl'], dBatch['res'])

            resmask = torch.ne(dBatch['res'][:,1:], args.mask_id)
            loss = torch.sum(loss) / torch.sum(resmask)

            '''dBatch = tdBatch['rule']
            if args.use_torch_amp:
                with torch_amp.autocast():
                    loss2, _ = model(dBatch['nl'], dBatch['nlparent'], dBatch['res'], dBatch['parent'], lefttree=True)
            else:
                loss2, _ = model(dBatch['nl'], dBatch['nlparent'], dBatch['res'], dBatch['parent'], lefttree=True)
            resmask = torch.ne(dBatch['res'][:,1:], args.mask_id)
            loss2 = torch.sum(loss2) / torch.sum(resmask)
            loss3 = 0.5 * loss + 0.5 * loss2'''
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
                if j % args.gradient_accumulation_steps == 0:
                    #loss.backward()
                    accelerator.backward(loss)
                else:
                    with accelerator.no_sync(model):
                        accelerator.backward(loss)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    accelerator.log({"loss_real": loss.item() * args.gradient_accumulation_steps, 'epoch':epoch}, step=global_step)
                accelerator.log({"loss":loss.item() * args.gradient_accumulation_steps})
            j += 1


def pretrain2():
    args.batch_size = 32 * 6 * 5
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='bf16', log_with='wandb')
    hps = {"num_iterations": 100, "learning_rate": 2e-4}

    accelerator.init_trackers("grammar-t5", config=hps)
    torch.manual_seed(args.seed)        
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_set = SumDataset(args, None, "train", idx=accelerator.process_index, mode='gen')        
    device = accelerator.device
    totoalnumber = len(train_set) * accelerator.num_processes
    args.rulenum = train_set.rulenum
    model = Decoder(args)
    load_model(model, 'checkpointEpchLR99Iter10010/')
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=50 * totoalnumber // args.batch_size)
    pathnames = []

    global_step = 0
    train_size = args.batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_size
    model, optimizer = accelerator.prepare(model, optimizer)    
    print(accelerator.num_processes)

    accelerator.register_for_checkpointing(model)
    accelerator.register_for_checkpointing(optimizer)
    accelerator.register_for_checkpointing(scheduler)
    for epoch in range(50):
        j = 0    
        sampler = ChunkedRandomSampler(train_set, train_size)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=15,collate_fn=rs_collate_fn1, sampler=sampler, pin_memory=True)
        for dBatch in tqdm(data_loader):
            if j % 10000 == 1000 and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                if len(pathnames) < 10:
                    save_model(unwrapped_model, 'checkpointEpch%dIter%d/'%(epoch, j))#print(loss.item())
                    #accelerator.save_state('checkpointEpch%dIter%d/'%(epoch, j))
                    pathnames.append('checkpointEpch%dIter%d/'%(epoch, j))
                else:
                    os.system('rm -r %s' % pathnames[0])
                    pathnames.pop(0)
                    save_model(unwrapped_model, 'checkpointEpch%dIter%d/'%(epoch, j))#print(loss.item())
                    #accelerator.save_state('checkpointEpch%dIter%d/'%(epoch, j))
                    pathnames.append('checkpointEpch%dIter%d/'%(epoch, j))
            model.train()
            for x in dBatch:
                dBatch[x] = dBatch[x].to(device)
            #iden
            if args.use_torch_amp:
                with torch_amp.autocast():
                    loss, _ = model(dBatch['nl'], dBatch['nlparent'], dBatch['res'], dBatch['parent'], lefttree=False)
            else:
                loss, _ = model(dBatch['nl'], dBatch['res'])

            resmask = torch.ne(dBatch['res'][:,1:], args.mask_id)
            loss3 = torch.sum(loss) / torch.sum(resmask)

            if loss3.item() == np.inf:
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
                if j % args.gradient_accumulation_steps == 0:
                    accelerator.backward(loss)
                else:
                    with accelerator.no_sync(model):
                        accelerator.backward(loss)
                if j % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()            
                    global_step += 1
                    accelerator.log({"loss_real": loss.item() * args.gradient_accumulation_steps, 'lr':scheduler.get_lr()[0], 'epoch':epoch})
                accelerator.log({"loss":loss.item() * args.gradient_accumulation_steps})
            j += 1

def testone():
    #pre()
    lang = 'java'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev_set = SumDataset(args, dataName="testone")
    args.rulenum = dev_set.rulenum
    model = Decoder(args)
    load_model(model, "checkpointEpch48Iter1000/")        
    if use_cuda:#torch.cuda.is_available():
        print('using GPU')
        model = model.cuda()
    while True:
        s = input('please input:')#"open a file named filename, and read the lines of the file into a string named line."
        dev_set.processOne(s)
        device = 'cuda:0'
        model = model
        args.batch_size = 1
        data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                    shuffle=False, drop_last=False, num_workers=0, collate_fn=rs_collate_fn1)
        model = model.eval()
        from beamsearch import BeamSearch
        from stringfy import strfy
        beamsize = 5
        beam = BeamSearch(beamsize, dev_set.ruledict, 2)
        for dBatch in tqdm(data_loader):
            dBatch['nl'] = dBatch['nl'].to(device).repeat(beamsize, 1)
            ans = beam.search(dBatch['nl'], model, lang=lang, max_len=512)        
            for i in range(len(ans)):
                for j in range(len(ans[i].set)):
                    beamone = ans[i].set[j]
                    root = beam.convertrulelist2tree(beamone.state, lang)
                    #print(root.printTree(root))
                    print(strfy(root.printTree(root), lang))
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
    args.batch_size = 10
    print(len(dev_set))
    devloader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=0, collate_fn=rs_collate_fn)
    model = model.eval()
    #load_model(model)
    f = open("outval1.txt", "w")
    index = 0 
    for x in tqdm(devloader):
        if index <= 0:
            index += 1
            continue
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
import line_profiler
def finetune():
    args.gradient_accumulation_steps = 1    
    args.pretrain_name = "grammart5-base"
    taskconfig = json.loads(open('processdata/data/%s/config.json'%args.task, 'r').read())
    args.NlLen = taskconfig["NlLen"]
    args.CodeLen = taskconfig["CodeLen"]
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=taskconfig['precision'], log_with='wandb')    
    args.batch_size = taskconfig["batch_size"][args.pretrain_name] * 1 * accelerator.num_processes
    if accelerator.is_main_process:
        data = pickle.load(open("processdata/%strain.pkl"%args.task, "rb"))
        datalen = len(data) // accelerator.num_processes
        print(datalen)
        for i in range(accelerator.num_processes):
            pickle.dump(data[i * datalen : (i + 1) * datalen], open("fttrain%d.pkl"%(i), "wb"))
        data = pickle.load(open("processdata/%svalid.pkl"%args.task, "rb"))
        if args.task in ['test']:
            data = data[:5000]
        if len(data) % accelerator.num_processes != 0:
            datalen = len(data) // accelerator.num_processes + 1
        else:
            datalen = len(data) // accelerator.num_processes
        for i in range(accelerator.num_processes):
            pickle.dump(data[i * datalen : (i + 1) * datalen], open("ftvalid%d.pkl"%(i), "wb"))
    newruledic = pickle.load(open("processdata/%srules.pkl"%args.task, "rb"))
    nrulelen = len(newruledic)
    accelerator.wait_for_everyone()
    hps = {"num_iterations": 100, "learning_rate": taskconfig['lr'], 'bs':taskconfig["batch_size"]}

    accelerator.init_trackers(args.task, config=hps)
    torch.manual_seed(args.seed + accelerator.process_index)        
    np.random.seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)
    train_set = SumDataset(args, None, "train", idx=accelerator.process_index, mode='finetune')   
    device = accelerator.device
    args.rulenum = len(train_set.ruledict)
    totoalnumber = len(train_set) * accelerator.num_processes
    model = Decoder(args)
    #model.encoder.model.resize_token_embeddings(args.rulenum)
    #model.tie_word_embeddings()
    #load_model(model, 'checkpointEpchLR99Iter30010/')    
    load_model(model, '%s-model/'%args.pretrain_name)    
    args.rulenum = nrulelen
    model.resize_token_embeddings(nrulelen)
    optimizer = optim.AdamW(model.parameters(), eps=1e-8, lr=taskconfig["lr"])
    from transformers import get_linear_schedule_with_warmup
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=30 * totoalnumber // args.batch_size)
    pathnames = []

    global_step = 0
    train_size = args.batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = train_size

    model, optimizer = accelerator.prepare(model, optimizer)    
    accelerator.register_for_checkpointing(model)
    accelerator.register_for_checkpointing(optimizer)
    #accelerator.register_for_checkpointing(scheduler)
    num_trial = patience = 0
    isBetter = False
    #optimizer = ScheduledOptim(optimizer, d_model=args.embedding_size, max_steps=50000)
    maxAcc= 0
    maxC = 0
    minloss = 1e10
    global_step = 0
    avgruntime = 0        
    dev_set = SumDataset(args, None, "valid", idx=accelerator.process_index)    
    test_set = SumDataset(args, None, "test", idx=accelerator.process_index)    

    if accelerator.is_main_process:
        open('communicate.txt', 'w').write('0')

    if args.eval:
        load_model(model.module, 'checkModel%s/'%args.task)    
    for epoch in range(1000):
        j = 0
        sampler = ChunkedRandomSampler(train_set, train_size)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=10,collate_fn=rs_collate_fn1, sampler=sampler, pin_memory=True)
        for dBatch in tqdm(data_loader):
            isBetter = False
            if j % 400 == 0 and epoch != 0 or args.eval:                    
                accelerator.wait_for_everyone()
                if args.task in ['transc2j', 'transj2c', 'assert', 'repair', 'repairme']: 
                    tnum, belu = evalmodelacc(dev_set, model, device, accelerator)
                    tnumtest, belutest = evalmodelacc(test_set, model, device, accelerator)

                else:
                    if args.task in ['django', 'conala', 'hs', 'mbpp']:
                        tnum, belu = evalmodel(dev_set, model, device, accelerator, newruledic, 'python')
                    elif args.task in ['commentjava', 'commentpython']:
                        tnum, belu = evalmodelnl(dev_set, model, device, accelerator)
                    elif args.task in ['transj2c']:
                        tnum, belu = evalmodel(dev_set, model, device, accelerator, newruledic, 'csharp')
                    else:
                        tnum, belu = evalmodel(dev_set, model, device, accelerator, newruledic, 'java')
                if accelerator.is_main_process:

                    open('communicate.txt', 'w').write('0')
                    if args.eval:
                        print(taskconfig["metric"])
                        print('current acc and num %f %f'%(belu, tnum))
                        print('current acc and num %f %f'%(belutest, tnumtest))

                        exit(0)
                    accelerator.log({"dev_bleu": belu, "dev_num": tnum, 'patience':patience, "trial":num_trial})
                    print('current acc and num %f %f'%(belu, tnum))
                    if maxC < tnum:
                        maxC = tnum
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(unwrapped_model, 'checkModelNUM/')
                        if taskconfig["metric"] == "acc" or args.task == 'concode':
                            isBetter = True
                        #print('find better accuracy %f'%tnum)
                        #save_model(model)
                    if maxAcc <= belu:
                        if taskconfig["metric"] == "bleu":
                            isBetter = True
                        if False:#args.task in ['repair', 'repairme']:
                            if num_trial > 3:
                                isBetter = True
                                maxAcc = belu
                        else:
                            maxAcc = belu
                    if isBetter:
                        patience = 0                        
                        print('find better acc %d'%(tnum))
                        print('save model to [%s]' % 'checkModel%s/'%args.task, file=sys.stderr)
                        #save_model(model.module, 'checkModel%d-%d/'%(epoch, j))
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(unwrapped_model, 'checkModel%s/'%args.task)
                        #accelerator.save_state('checkpoint/')
                        os.system('cp out.txt out1.txt')

                    elif patience < args.patience:
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)
                        if patience == args.patience:
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == args.max_num_trials:
                                print('early stop!', file=sys.stderr)
                                exit(0)
                            #lr = optimizer.param_groups[0]['lr'] * 0.5
                            open('communicate.txt', 'w').write('1')
                            #accelerator.load_state('checkpoint/')
                            patience = 0
                    else:
                        patience += 1
                    save_model(accelerator.unwrap_model(model), 'checkModelLast%s/'%args.task)
                accelerator.wait_for_everyone()                    
                reloads = open('communicate.txt', 'r').read()
                if reloads == '1':
                    load_model(model.module, 'checkModel%s/'%args.task)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.5 * param_group['lr']                    
                    print('reload')
                accelerator.wait_for_everyone()
            model.train()
            for x in ((dBatch)):
                dBatch[x] = dBatch[x].to(device)
            starttime = time.time()
            loss, _ = model(dBatch['nl'], dBatch['res'])
            avgruntime += time.time() - starttime
            resmask = torch.ne(dBatch['res'][1:], args.mask_id)
            loss = torch.sum(loss) / torch.sum(resmask)
            if loss.sum().item() == np.inf:
                print('inf')
                exit(0)
            if loss.item() == np.inf:
                print(j)
                assert(0)
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            if j % args.gradient_accumulation_steps == 0:
                accelerator.backward(loss)
            else:
                with accelerator.no_sync(model):
                    accelerator.backward(loss)
            if j % args.gradient_accumulation_steps == 0:
                optimizer.step()#_and_update_lr()
                optimizer.zero_grad()
                #scheduler.step()
            accelerator.log({"loss": loss.item()})
            #wandb.log({"loss": loss.item()})
            j += 1
            global_step += 1
        accelerator.log({"runtime": avgruntime / j})
            #display("metrics")
            #display("assets")
@torch.no_grad()
def evalmodel(dev_set, model, device, accelerator, newruledic, lang):
    data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=25, drop_last=False, num_workers=2,collate_fn=rs_collate_fn1, shuffle=False, pin_memory=True)
    from beamsearch import BeamSearch
    beamsize = 3
    beam = BeamSearch(beamsize, newruledic, 1)
    model.eval()
    f = open("outval%d.txt"%int(accelerator.process_index), "w")
    for dBatch in tqdm(data_loader):
        dBatch['nl'] = dBatch['nl'].to(device).repeat_interleave(beamsize, dim=0)
        ans = beam.search(dBatch['nl'], model, lang=lang, max_len=args.CodeLen, vocabsize=args.rulenum)        
        for i in range(len(ans)):
            beamone = ans[i].set[0]
            root = beam.convertrulelist2tree(beamone.state, lang)
            f.write(root.printTree(root))
            f.write("\n")
            f.write(str(beamone.state) + "\n")
    f.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.system("python3 sum.py %d %s %s"% (accelerator.num_processes, lang, args.task))
        from evaluator.CodeBLEU import calc_code_bleu
        if lang == 'python':
            testlang = 'java'
        else:
            testlang = lang.replace('csharp', 'c_sharp')
        tnum, codebelu = calc_code_bleu.get_codebleu("processdata/groundvalid%s.txt"%args.task, "out.txt", testlang, benchmark=args.task)
        return tnum, codebelu
    else:
        return 0, 0
@torch.no_grad()
def evalmodelacc(dev_set, model, device, accelerator):
    data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=25, drop_last=False, num_workers=2,collate_fn=rs_collate_fn1, shuffle=False, pin_memory=True)
    accs = []
    tnums = []
    model.eval()
    deta = []
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
        deta.extend(tnum.tolist())
        tnums.append(tnum.sum().item())
        
    tnum = np.sum(tnums)
    acc = np.mean(accs)
    open("resdetail%d.txt"%int(accelerator.process_index), "w").write(str(deta))
    open("res%d.txt"%int(accelerator.process_index), "w").write(str(acc) + "\t" + str(tnum))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accs = []
        tnums = []
        for i in range(accelerator.num_processes):
            acc, tnum = open("res%d.txt"%i).read().split()
            accs.append(float(acc))
            tnums.append(int(tnum))
        return np.sum(tnums), np.mean(accs)
    else:
        return 0, 0
@torch.no_grad()
def evalmodelnl(dev_set, model, device, accelerator):
    data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=25, drop_last=False, num_workers=2,collate_fn=rs_collate_fn1, shuffle=False, pin_memory=True)
    from beamsearch import BeamSearch
    beamsize = 1
    beam = BeamSearch(beamsize, dev_set.ruledict, 0)
    model.eval()
    f = open("outval%d.txt"%int(accelerator.process_index), "w")
    for dBatch in tqdm(data_loader):
        dBatch['nl'] = dBatch['nl'].to(device).repeat_interleave(beamsize, dim=0)
        ans = beam.search(dBatch['nl'], model, max_len=args.CodeLen, vocabsize=args.rulenum, mode='nl')        
        for i in range(len(ans)):
            beamone = ans[i].set[0]
            root = beam.convertrulelist2tree(beamone.state, mode='nl')
            f.write(root)
            f.write("\n")
            f.flush()
    f.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        preds = []
        output = open('out.txt', 'w')
        for i in range(accelerator.num_processes):
            f = open("outval%d.txt"%i, "r")
            for line in f:
                preds.append(str(len(preds)) + '🚀' + line.strip())
                output.write(line.strip() + '\n')
            f.close()
        output.close()
        f = open("outgr.txt", "w")
        lines = open("processdata/groundvalid%s.txt"%args.task, "r").readlines()
        for i in range(len(preds)):
            f.write(str(i) + '🚀' + lines[i].strip() + '\n')
        from bleunl import calbleu
        bleu = calbleu('outgr.txt', preds)
        return 0, bleu
    else:
        return 0, 0
def testfill():
    lang = 'java'
    args.NlLen = 350
    args.CodeLen = 256
    rule = pickle.load(open('csharprule.pkl', 'rb'))
    args.rulenum = len(rule)
    model = Decoder(args).to('cuda:0')
    load_model(model, 'checkpointEpchLR47Iter10010/')    
    model.eval()
    testcode = '''  private boolean isInlinableObject(List<Reference> refs) {
      boolean ret = false;
      Set<String> validProperties = Sets.newHashSet();
      for (Reference ref : refs) {
        Node name = ref.getNode();
        Node parent = ref.getParent();
        Node gramps = ref.getGrandparent();

        // Ignore most indirect references, like x.y (but not x.y(),
        // since the function referenced by y might reference 'this').
        //
        if (parent.isGetProp()) {
          Preconditions.checkState(parent.getFirstChild() == name);
          // A call target may be using the object as a 'this' value.
          if (gramps.isCall()
              && gramps.getFirstChild() == parent) {
            return false;
          }

          // Deleting a property has different semantics from deleting
          // a variable, so deleted properties should not be inlined.
          extra_id_0;
          // NOTE(nicksantos): This pass's object-splitting algorithm has
          // a blind spot. It assumes that if a property isn't defined on an
          // object, then the value is undefined. This is not true, because
          // Object.prototype can have arbitrary properties on it.
          //
          // We short-circuit this problem by bailing out if we see a reference
          // to a property that isn't defined on the object literal. This
          // isn't a perfect algorithm, but it should catch most cases.
          String propName = parent.getLastChild().getString();
          if (!validProperties.contains(propName)) {
            if (NodeUtil.isVarOrSimpleAssignLhs(parent, gramps)) {
              validProperties.add(propName);
            } else {
              return false;
            }
          }
          continue;
        }

        // Only rewrite VAR declarations or simple assignment statements
        if (!isVarOrAssignExprLhs(name)) {
           return false;
        }

        Node val = ref.getAssignedValue();
        if (val == null) {
          // A var with no assignment.
          continue;
        }

        // We're looking for object literal assignments only.
        if (!val.isObjectLit()) {
          return false;
        }

        // Make sure that the value is not self-referential. IOW,
        // disallow things like x = {b: x.a}.
        //
        // TODO: Only exclude unorderable self-referential
        // assignments. i.e. x = {a: x.b, b: x.a} is not orderable,
        // but x = {a: 1, b: x.a} is.
        //
        // Also, ES5 getters/setters aren't handled by this pass.
        for (Node child = val.getFirstChild(); child != null;
             child = child.getNext()) {
          if (child.isGetterDef() ||
              child.isSetterDef()) {
            // ES5 get/set not supported.
            return false;
          }

          validProperties.add(child.getString());

          Node childVal = child.getFirstChild();
          // Check if childVal is the parent of any of the passed in
          // references, as that is how self-referential assignments
          // will happen.
          for (Reference t : refs) {
            Node refNode = t.getParent();
            while (!NodeUtil.isStatementBlock(refNode)) {
              if (refNode == childVal) {
                // There's a self-referential assignment
                return false;
              }
              refNode = refNode.getParent();
            }
          }
        }


        // We have found an acceptable object literal assignment. As
        // long as there are no other assignments that mess things up,
        // we can inline.
        ret = true;
      }
      return ret;
    }
'''
    from processdata import solvedata
    root = solvedata.parserTree([{'nl':"", "function":testcode}])
    print(root[0]['root'])
    from processdata import solvetree
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
    action, rules = solvetree.processaction(root)
    rulelist = [tokenizer.cls_token_id] + [tokenizer.sep_token_id] + action[0]['rulelist'][1:-1] + [tokenizer.sep_token_id] 
    inputnl = torch.tensor([rulelist]).to('cuda:0')
    from beamsearch import BeamSearch
    beam = BeamSearch(10, rule, 1)
    inputnl = inputnl.repeat_interleave(10, dim=0)
    ans = beam.search(inputnl, model, lang=lang, max_len=args.CodeLen, vocabsize=args.rulenum, mode='fill')        
    for i in range(len(ans[0].set)):
        beamone = ans[0].set[i]
        root = beam.convertrulelist2tree(beamone.state, lang, mode='fill')
        print(root.printTree(root))
if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='pretrain', type=str, required=True)
    argc = parser.parse_args()
    args.task = argc.dataset
    if args.task in ['django', 'concode', 'codetrans', 'repair', 'assert', 'conala', 'hs', 'test', 'repairme', 'transj2c', 'transc2j', 'commentjava', 'commentpython', 'mbpp']:
        args.eval = False
        finetune()
    if args.task in ['pretrain']:
        pretrain()
    if args.task in ['pretrain2']:
        pretrain2()
    if args.task in ['fill']:
        testfill()
    if args.task in ['searchadv', 'searchcos']:
        #args.eval = False
        from runsearch import finetune_search
        finetune_search(args)
    #pretrain2()
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
