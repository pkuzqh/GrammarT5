import torch.nn as nn
import torch.nn.functional as F
import torch
from Transfomer import TransformerBlock
from relTransformer import rightTransformerBlock
from Multihead_Combination import MultiHeadedCombination
from Embedding import Embedding
from TreeConvGen import TreeConvGen
from Multihead_Attention import MultiHeadedAttention
from gelu import GELU
from LayerNorm import LayerNorm
from decodeTrans import decodeTransformerBlock
from gcnnnormal import GCNNM
from torch.nn.parameter import Parameter
import pickle
from torch.nn.parameter import Parameter
from postionEmbedding import PositionalEmbedding
from graphTransfomer import graphTransformerBlock
from transformers import AutoModel
from Grape import Grape
from FastAttention import FastMultiHeadedAttention
from fastTransformer import fastTransformerBlock
from RelEmbedding import RelEmbeddings
class NlEncoder1(nn.Module):
    def __init__(self, args):
        super(NlEncoder1, self).__init__()
        self.embedding_size = args.embedding_size
        self.mask_id = args.mask_id
        self.feed_forward_hidden = 4 * self.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.model = nn.ModuleList([TransformerBlock(self.embedding_size, 12, self.feed_forward_hidden, 0.1) for _ in range(12)])
        self.embeddings = nn.Embedding(args.bertnum + 10, self.embedding_size)

    def forward(self, input_nl):
        nlmask = torch.ne(input_nl, self.mask_id)
        inputnlem = self.embeddings(input_nl)
        encode = inputnlem
        for x in self.model:
            encode = x(encode, nlmask)
        return encode, nlmask
class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.mask_id = args.mask_id
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.model = AutoModel.from_pretrained('Salesforce/codet5-small').encoder

    def forward(self, input_nl):
        nlmask = torch.ne(input_nl, self.mask_id)
        #print(input_nl)
        encode = self.model(input_nl, attention_mask=nlmask)
        encode = encode.last_hidden_state
        return encode, nlmask
    def getEm(self):
        return self.model.embed_tokens 
from transformers import T5EncoderModel, AutoConfig, T5ForConditionalGeneration
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        config = AutoConfig.from_pretrained(args.pretrain_name)
        self.model = AutoModel.from_config(config)#pretrained(args.pretrain_name)
        args.embedding_size = self.model.config.hidden_size
        self.mask_id = args.mask_id
        self.lm_head = nn.Linear(args.embedding_size, args.rulenum, bias=False)
        self.embedding_size = args.embedding_size
        self.vocab_size = args.rulenum
        self.model.encoder.resize_token_embeddings(self.vocab_size)
        self.model.set_input_embeddings(self.model.encoder.embed_tokens)
        self.lm_head.weight = self.model.decoder.embed_tokens.weight
    def nl_encode(self, inputnl):
        nlmask = torch.ne(inputnl, self.mask_id)
        encoder_outputs = self.model.encoder(inputnl, attention_mask=nlmask)
        return encoder_outputs.last_hidden_state, nlmask
    def resize_token_embeddings(self, new_num_tokens):
        self.model.encoder.resize_token_embeddings(new_num_tokens)
        self.model.set_input_embeddings(self.model.encoder.embed_tokens)
        self.lm_head.weight = self.model.decoder.embed_tokens.weight
    def forward(self, inputnl, inputrule, mode="train"):
        inputRes = inputrule[:, 1:].long()
        inputrule = inputrule[:, :-1].long()
        rulemask = torch.ne(inputrule, self.mask_id)
        nlmask = torch.ne(inputnl, self.mask_id)
        encoder_outputs = self.model.encoder(inputnl.long(), attention_mask=nlmask)
        hidden_states = encoder_outputs.last_hidden_state
        ouput = self.model.decoder(inputrule, attention_mask=rulemask, encoder_hidden_states=hidden_states, encoder_attention_mask=nlmask)
        ouput = ouput.last_hidden_state
        #tie-word-embedding
        ouput = ouput * (self.embedding_size**-0.5)
        resSoftmax = torch.softmax(self.lm_head(ouput), dim=-1)
        if mode != "train":
            return resSoftmax
        resmask = torch.ne(inputRes, self.mask_id)
        loss = -torch.log(torch.gather(resSoftmax, -1, inputRes.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(resmask == 0, 0.0)
        resTruelen = torch.sum(resmask, dim=-1).float()
        return loss, resSoftmax  
    def encode_nl(self, inputnl):
        nlmask = torch.ne(inputnl, self.mask_id)
        encoder_outputs = self.model.encoder(inputnl, attention_mask=nlmask)
        return encoder_outputs.last_hidden_state, nlmask
    def test_forward(self, nlencode, nlmask, inputrule, past_key_values=None):
        rulemask = torch.ne(inputrule, self.mask_id)
        ouput = self.model.decoder(inputrule, attention_mask=None, encoder_hidden_states=nlencode, encoder_attention_mask=nlmask, past_key_values=past_key_values)
        past_key_values = ouput.past_key_values
        ouput = ouput.last_hidden_state
        #tie-word-embedding
        ouput = ouput * (self.embedding_size**-0.5)
        resSoftmax = torch.softmax(self.lm_head(ouput), dim=-1)            
        return resSoftmax, past_key_values
from transformers import T5ForConditionalGeneration 
class Decoder1(nn.Module):
    def __init__(self, args):
        super(Decoder1, self).__init__()
        self.model =  T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
        self.mask_id = args.mask_id
    def forward(self, inputids, outputids):
        decoderinputid =  outputids[:, :-1].long()
        decoderoutputid = outputids[:, 1:].long()
        inputmask = torch.ne(inputids, self.mask_id)
        outputmask = torch.ne(decoderinputid, self.mask_id)
        output = self.model(input_ids=inputids, attention_mask=inputmask, decoder_input_ids=decoderinputid, decoder_attention_mask=outputmask)
        resSoftmax = torch.softmax(output.logits, dim=-1)
        resmask = torch.ne(decoderoutputid, self.mask_id)
        loss = -torch.log(torch.gather(resSoftmax, -1, decoderoutputid.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(resmask == 0, 0.0)
        return loss, resSoftmax



class searchModel(nn.Module):
    def __init__(self, model):
        super(searchModel, self).__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
    def forward(self, input_ids):
        attention_mask = torch.ne(input_ids, self.model.mask_id)
        decoder_ids = self.model.model._shift_right(input_ids)
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask,
                               decoder_input_ids=decoder_ids, decoder_attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        eos_mask = input_ids.eq(self.model.model.config.eos_token_id)
        #print(self.model.model.config.eos_token_id, input_ids)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            eos_mask = torch.sum(eos_mask, 1)
            #print(eos_mask)
            #print(input_ids[3])
            raise ValueError("All examples must have the same number of <eos> tokens.")
        #print(hidden_states.size())
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        #print(vec.size())
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec
    def forward1(self, input_ids):
        attention_mask = torch.ne(input_ids, self.model.mask_id)
        outputs = self.model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        outputs = (hidden_states*attention_mask[:,:,None]).sum(1)/attention_mask.sum(-1)[:,None]
        vec = torch.nn.functional.normalize(outputs, p=2, dim=1)
        return vec

    def cal_loss(self, nl_encode, code_encode):
        bs = nl_encode.size(0)
        scores=(nl_encode[:,None,:]*code_encode[None,:,:]).sum(-1)
        loss = self.loss(20 * scores, torch.arange(bs, device=scores.device))
        return loss





