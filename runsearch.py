from accelerate import Accelerator
from Model import searchModel, Decoder
import torch
import pickle
import numpy as np
import random
from Dataset import *
from torch import optim
import time
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
def finetune_search(args):
    args.gradient_accumulation_steps = 1
    args.NlLen = 128
    args.CodeLen = 312
    if args.task in ['comment', 'searchadv', 'searchcos']:
        args.embedding_size = 768
        args.pretrain_name = 'Salesforce/codet5-base'
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision='bf16', log_with='wandb')    
    args.batch_size = 30 * 1 * accelerator.num_processes

    if accelerator.is_main_process:
        data = pickle.load(open("processdata/%strain.pkl"%args.task, "rb"))
        datalen = len(data) // accelerator.num_processes
        print(datalen)
        for i in range(accelerator.num_processes):
            pickle.dump(data[i * datalen : (i + 1) * datalen], open("fttrain%d.pkl"%(i), "wb"))
        data = pickle.load(open("processdata/%svalid.pkl"%args.task, "rb"))
        if len(data) % accelerator.num_processes != 0:
            datalen = len(data) // accelerator.num_processes + 1
        else:
            datalen = len(data) // accelerator.num_processes
        for i in range(accelerator.num_processes):
            pickle.dump(data[i * datalen : (i + 1) * datalen], open("ftvalid%d.pkl"%(i), "wb"))
        data = pickle.load(open("processdata/%stestbase.pkl"%args.task, "rb"))
        if len(data) % accelerator.num_processes != 0:
            datalen = len(data) // accelerator.num_processes + 1
        else:
            datalen = len(data) // accelerator.num_processes
        for i in range(accelerator.num_processes):
            pickle.dump(data[i * datalen : (i + 1) * datalen], open("fttestbase%d.pkl"%(i), "wb"))
    nrulelen = len(pickle.load(open("processdata/%srules.pkl"%args.task, "rb")))
    accelerator.wait_for_everyone()
    hps = {"num_iterations": 100, "learning_rate": 3e-5}

    accelerator.init_trackers(args.task, config=hps)
    torch.manual_seed(args.seed + accelerator.process_index)        
    np.random.seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)
    train_set = SumDataset(args, None, "train", idx=accelerator.process_index, mode='finetunesearch')   
    device = accelerator.device
    args.rulenum = len(train_set.ruledict)
    totoalnumber = len(train_set) * accelerator.num_processes
    model = Decoder(args)
    #model.encoder.model.resize_token_embeddings(args.rulenum)
    #model.tie_word_embeddings()
    load_model(model, 'checkpointEpchLR99Iter30010/')    
    args.rulenum = nrulelen
    model.resize_token_embeddings(nrulelen)
    optimizer = optim.AdamW(model.parameters(), eps=1e-8, lr=2e-5)
    from transformers import get_linear_schedule_with_warmup
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=30 * totoalnumber // args.batch_size)
    pathnames = []
    model = searchModel(model)
    #load_model(model, "checkModel/")
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
    dev_set_base = SumDataset(args, None, "testbase", idx=accelerator.process_index)    
    if accelerator.is_main_process:
        open('communicate.txt', 'w').write('0')

    if args.eval:
        load_model(model.module, 'checkModel/')
        data = pickle.load(open("processdata/%stest.pkl"%args.task, "rb"))
        if len(data) % accelerator.num_processes != 0:
            datalen = len(data) // accelerator.num_processes + 1
        else:
            datalen = len(data) // accelerator.num_processes
        for i in range(accelerator.num_processes):
            pickle.dump(data[i * datalen : (i + 1) * datalen], open("fttest%d.pkl"%(i), "wb"))
        dev_set = SumDataset(args, None, "test", idx=accelerator.process_index)    

    for epoch in range(1000):
        j = 0
        sampler = ChunkedRandomSampler(train_set, train_size)
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_size, drop_last=True, num_workers=10,collate_fn=rs_collate_fn2, shuffle=True, pin_memory=True)
        for dBatch in tqdm(data_loader):
            isBetter = False
            if j  % 200 == 0 or args.eval:                    
                accelerator.wait_for_everyone()
                if args.task in ['searchadv', 'searchcos']: 
                    tnum, belu = evalmodel(args, dev_set_base, dev_set, model, device, accelerator)
                if accelerator.is_main_process:

                    open('communicate.txt', 'w').write('0')
                    if args.eval:
                        print('current acc and num %f %f'%(belu, tnum))
                        exit(0)
                    accelerator.log({"dev_bleu": belu, "dev_num": tnum})
                    print('current acc and num %f %f'%(belu, tnum))
                    if maxC < tnum:
                        maxC = tnum
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(unwrapped_model, 'checkModelNUM/')
                        #isBetter = True
                        #print('find better accuracy %f'%tnum)
                        #save_model(model)
                    if maxAcc < belu:
                        isBetter = True
                        maxAcc = belu
                        print('find better acc %f'%belu)
                    if isBetter:
                        patience = 0
                        print('save model to [%s]' % 'checkModel/', file=sys.stderr)
                        #save_model(model.module, 'checkModel%d-%d/'%(epoch, j))
                        unwrapped_model = accelerator.unwrap_model(model)
                        save_model(unwrapped_model, 'checkModel/')
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
                accelerator.wait_for_everyone()                    
                reloads = open('communicate.txt', 'r').read()
                if reloads == '1':
                    load_model(model.module, 'checkModel/')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 0.5 * param_group['lr']                    
                    print('reload')
                accelerator.wait_for_everyone()
            model.train()
            for x in ((dBatch)):
                dBatch[x] = dBatch[x].to(device)
            starttime = time.time()
            
            nlencode = model(dBatch['nl'])
            codeencode = model(dBatch['res'])
            loss = model.cal_loss(nlencode, codeencode)

            avgruntime += time.time() - starttime
            loss = torch.mean(loss)
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
def evalmodel(args, dev_set_base, dev_set, model, device, accelerator):
    data_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=25, drop_last=False, num_workers=2,collate_fn=rs_collate_fn2, shuffle=False, pin_memory=True)
    data_loader_base = torch.utils.data.DataLoader(dataset=dev_set_base, batch_size=25, drop_last=False, num_workers=2,collate_fn=rs_collate_fn2, shuffle=False, pin_memory=True)

    model.eval()
    nlencodelst = []
    codeencodelst = []
    #f = open("outval%d.txt"%int(accelerator.process_index), "w")
    for dBatch in tqdm(data_loader):
        dBatch['nl'] = dBatch['nl'].to(device)
        nlencode = model(dBatch['nl'])
        nlencodelst.append(nlencode.to(torch.float32).cpu().numpy())
    for dBatch in tqdm(data_loader_base):
        dBatch['res'] = dBatch['res'].to(device)
        codeencode = model(dBatch['res'])
        codeencodelst.append(codeencode.to(torch.float32).cpu().numpy())

    code_vecs = np.concatenate(codeencodelst,0)
    nl_vecs = np.concatenate(nlencodelst,0)
    open('codevecs%d.pkl'%accelerator.process_index, 'wb').write(pickle.dumps(code_vecs))
    open('nlvecs%d.pkl'%accelerator.process_index, 'wb').write(pickle.dumps(nl_vecs))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        codevecs = []
        nlvecs = []
        for i in range(accelerator.num_processes):
            nlvecs.append(pickle.load(open('nlvecs%d.pkl'%i, 'rb')))
            codevecs.append(pickle.load(open('codevecs%d.pkl'%i, 'rb')))
        codevecs = np.concatenate(codevecs, 0)
        nlvecs = np.concatenate(nlvecs, 0)
        scores = np.matmul(nlvecs,codevecs.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]  
        urls = pickle.load(open('processdata/%surl.pkl'%args.task, 'rb'))
        code_urls = urls['code']
        nl_urls = urls['nl']
        ranks = []

        for i, sort_id in enumerate(sort_ids):
            rank = 0
            find = False
            for idx in sort_id[:1000]:
                if find is False:
                    rank += 1
                if code_urls[idx] == nl_urls[i]:
                    find = True
            if find:
                ranks.append(1/rank)
            else:
                ranks.append(0)
        return 0, np.mean(ranks)
    else:
        return 0, 0