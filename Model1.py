import torch.nn as nn
import torch.nn.functional as F
import torch
from Transfomer import TransformerBlock
from rightTransformer import rightTransformerBlock
from Multihead_Combination import MultiHeadedCombination
from Embedding import Embedding
from TreeConvGen import TreeConvGen
from Multihead_Attention import MultiHeadedAttention
from gelu import GELU
from LayerNorm import LayerNorm
from decodeTrans import decodeTransformerBlock
from gcnnnormal import GCNNM
from torch.nn.parameter import Parameter
from run import gVar
import pickle
from torch.nn.parameter import Parameter
from postionEmbedding import PositionalEmbedding
from graphTransfomer import graphTransformerBlock
from transformers import AutoModel
from Grape import Grape
from FastAttention import FastMultiHeadedAttention
from fastTransformer import fastTransformerBlock
from RelEmbedding import RelEmbeddings
class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.mask_id = args.mask_id
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.model = AutoModel.from_pretrained('distilroberta-base')

    def forward(self, input_nl):
        nlmask = torch.ne(input_nl, self.mask_id)
        encode = self.model(input_nl, attention_mask=nlmask)[0]
        return encode, nlmask
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.cnum = args.cnum
        self.word_len = args.WoLen
        self.nl_len = args.NlLen
        self.code_len = args.CodeLen
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, args.WoLen))
        self.path_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        self.rule_conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 2))
        self.resLen = args.rulenum
        self.encodeTransformerBlock = nn.ModuleList(
            [fastTransformerBlock(self.embedding_size, 12, self.feed_forward_hidden, 0.1) for _ in range(12)])

        self.mask_id = args.mask_id
        self.resLinear = nn.Linear(self.embedding_size, self.resLen)
        self.rule_token_embedding = nn.Embedding(args.Code_Vocsize, self.embedding_size)
        self.rule_embedding = nn.Embedding(args.rulenum - args.bertnum, self.embedding_size)
        self.encoder = NlEncoder(args)
        self.layernorm = LayerNorm(self.embedding_size)
        self.ruleem = Grape(args, "pythonrule.pkl")
        self.dropout = nn.Dropout(p=0.1)
        self.max_rel_pos = args.max_rel_pos
        self.par_heads = 8
        self.bro_heads = 12 - self.par_heads
        self.end_nodes = None
        self.par_rel_emb = RelEmbeddings(self.embedding_size // 12, self.par_heads, self.max_rel_pos, 'p2q,p2k,p2v')
        self.bro_rel_emb = RelEmbeddings(self.embedding_size // 12, self.bro_heads, self.max_rel_pos, 'p2q,p2k,p2v')
    def nl_encode(self, input_nl):
        nlencode, nlmask = self.encoder(input_nl)
        return nlencode, nlmask
    def test_foward(self, nlencode, nlmask, inputrule, inputrulechild, inputParent, inputParentPath, tmpindex, tmpf, tmpc, tmpchar, tmpindex2, rulead, antimask, inputRes=None, mode="train"):
        selfmask = antimask
        rulemask = torch.ne(inputrule, self.mask_id)

        charEm = self.char_embedding(tmpchar.long())
        charEm = self.conv(charEm.permute(0, 3, 1, 2))
        charEm = charEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        rule_token_embedding = self.rule_token_embedding(tmpindex2[0])
        rule_token_embedding = rule_token_embedding + charEm[0]

        inputParent = inputParent.float()
        degree = torch.sum(inputParent, dim=-1, keepdim=True).clamp(min=1e-6)

        degree = 1.0 / degree

        inputParent = degree * inputParent# * degree

        childEm = F.embedding(tmpc, rule_token_embedding)
        childEm = self.conv(childEm.permute(0, 3, 1, 2))
        childEm = childEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        childEm = self.layernorm(childEm)
        fatherEm = F.embedding(tmpf, rule_token_embedding)#self.rule_token_embedding(tmpf)
        ruleEmCom = self.rule_conv(torch.stack([fatherEm, childEm], dim=-2).permute(0, 3, 1, 2))
        ruleEmCom = self.layernorm(ruleEmCom.permute(0, 2, 3, 1).squeeze(dim=-2))
        x = self.rule_embedding(tmpindex[0])
        rulenoter = self.layernorm(x + ruleEmCom[0])
        ruleselect = torch.cat([self.encoder.model.embeddings.word_embeddings.weight.clone(), rulenoter.squeeze(0)], dim=0)#torch.cat([rulenoter, ruleter], dim=0)
        ruleEm = F.embedding(inputrule, ruleselect)#self.rule_embedding(inputrule)

        Ppath = F.embedding(inputrulechild, rule_token_embedding)
        ppathEm = self.path_conv(Ppath.permute(0, 3, 1, 2))
        ppathEm = ppathEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        ppathEm = self.layernorm(ppathEm)
        x = (ruleEm)

        for trans in self.encodeTransformerBlock:
            x = trans(x, selfmask, nlencode, nlmask, ppathEm, inputParent)
        decode = x
        #ppath
        inputParentAd = inputParent[:,1:,1:]
        inputParentAd = F.pad(inputParentAd, (0, 1, 0, 1), "constant", 0)
        Ppath = F.embedding(inputParentPath, rule_token_embedding)#self.rule_token_embedding(inputParentPath)
        ppathEm = self.path_conv(Ppath.permute(0, 3, 1, 2))
        ppathEm = ppathEm.permute(0, 2, 3, 1).squeeze(dim=-2)
        ppathEm = self.layernorm(ppathEm)
        x = (ppathEm)
        for trans in self.decodeTransformerBlocksP:
            x = trans(x, rulemask, decode, antimask, nlencode, nlmask, inputParentAd)
        decode = x

        genP2 = torch.softmax(self.resLinear((decode)), dim=-1)

        resSoftmax = genP2
        if mode != "train":
            return resSoftmax

        resmask = torch.ne(inputRes, self.mask_id)
        loss = -torch.log(torch.gather(resSoftmax, -1, inputRes.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(resmask == 0, 0.0)
        resTruelen = torch.sum(resmask, dim=-1).float()
        totalloss = torch.sum(loss, dim=-1)/ resTruelen
        totalloss = totalloss #+ (self.getBleu(loss, 2) + self.getBleu(loss, 3) + self.getBleu(loss, 4)) / resTruelen
        loss = totalloss #+ lossselect
        return loss, resSoftmax
    def concat_pos(self, rel_par_pos, rel_bro_pos):
        if self.par_heads == 0:
            return rel_bro_pos.unsqueeze(1).repeat_interleave(repeats=self.bro_heads,
                                                              dim=1)
        if self.bro_heads == 0:
            return rel_par_pos.unsqueeze(1).repeat_interleave(repeats=self.par_heads,
                                                              dim=1)

        rel_par_pos = rel_par_pos.unsqueeze(1).repeat_interleave(repeats=self.par_heads,
                                                                 dim=1)
        rel_bro_pos = rel_bro_pos.unsqueeze(1).repeat_interleave(repeats=self.bro_heads,
                                                                 dim=1)
        rel_pos = self.concat_vec(rel_par_pos, rel_bro_pos, dim=1)

        return rel_pos
    def concat_vec(self, vec1, vec2, dim):
        if vec1 is None:
            return vec2
        if vec2 is None:
            return vec1
        return torch.cat([vec1, vec2], dim=dim)
    def forward(self, inputnl, inputrule, inputParent, inputParentf, inputParenta, antimask, inputRes=None, mode="train"):
        selfmask = antimask
        rulemask = torch.ne(inputrule, self.mask_id)
        #start_endnodes
        need_end_nodes = True
        rel_par_pos = inputParentf
        rel_bro_pos = inputParenta
        batch_size, max_rel_pos, max_ast_len = rel_par_pos.size()
        start_nodes = self.concat_pos(rel_par_pos, rel_bro_pos)
        #if self.end_nodes is not None and batch_size == self.end_nodes.size(0):
        need_end_nodes = True
        if need_end_nodes:
            end_nodes = torch.arange(max_ast_len, device=start_nodes.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            end_nodes = end_nodes.repeat(batch_size, self.par_heads + self.bro_heads, max_rel_pos, 1)
        rel_par_q, rel_par_k, rel_par_v = self.par_rel_emb()
        rel_bro_q, rel_bro_k, rel_bro_v = self.bro_rel_emb()
        rel_q = torch.cat([rel_par_q, rel_bro_q], dim=1)
        rel_k = torch.cat([rel_par_k, rel_bro_k], dim=1)
        rel_v = torch.cat([rel_par_v, rel_bro_v], dim=1)
        #encode ruletoken
        rulenoter = self.layernorm(self.ruleem())
        ruleselect = torch.cat([self.encoder.model.embeddings.word_embeddings.weight.clone(), rulenoter.squeeze(0)], 
        dim=0)
        #encode rule
        ruleEm = F.embedding(inputrule, ruleselect)
        #encode nl
        nlencode, nlmask = self.encoder(inputnl)
        #encode path
        x = (ruleEm)
        #ast reader
        #mask = [inputParentf] * 6 + [inputParenta] * 6
        #mask = torch.stack(mask, dim=1)
        for trans in self.encodeTransformerBlock:
            #x = trans(x, selfmask, nlencode, nlmask, inputParent)
            #x = trans(x, inputParentf, nlencode, nlmask, inputParenta)
            x = trans(x, nlencode, nlmask, start_nodes, end_nodes, rel_q, rel_k, rel_v)
        decode = x
        genP2 = torch.softmax(self.resLinear((decode)), dim=-1)
        resSoftmax = genP2
        if mode != "train":
            return resSoftmax
        resmask = torch.ne(inputRes, self.mask_id)
        loss = -torch.log(torch.gather(resSoftmax, -1, inputRes.unsqueeze(-1)).squeeze(-1))
        loss = loss.masked_fill(resmask == 0, 0.0)
        resTruelen = torch.sum(resmask, dim=-1).float()
        return loss, resSoftmax



class JointEmbber(nn.Module):
    def __init__(self, args):
        super(JointEmbber, self).__init__()
        self.embedding_size = args.embedding_size
        self.codeEncoder = TreeAttEncoder(args)
        self.margin = args.margin
        self.nlEncoder = NlEncoder(args)
        self.poolConvnl = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.poolConvcode = nn.Conv1d(self.embedding_size, self.embedding_size, 3)
        self.maxPoolnl = nn.MaxPool1d(args.NlLen)
        self.maxPoolcode = nn.MaxPool1d(args.CodeLen)
    def scoring(self, qt_repr, cand_repr):
        sim = F.cosine_similarity(qt_repr, cand_repr)
        return sim
    def nlencoding(self, inputnl, inputnlchar):
        nl = self.nlEncoder(inputnl, inputnlchar)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        return nl
    def codeencoding(self, inputcode, inputcodechar, ad):
        code = self.codeEncoder(inputcode, inputcodechar, ad)
        code = self.maxPoolcode(self.poolConvcode(code.permute(0, 2, 1))).squeeze(-1)
        return code
    def forward(self, inputnl, inputnlchar, inputcode, inputcodechar, ad, inputcodeneg, inputcodenegchar, adneg):
        nl = self.nlEncoder(inputnl, inputnlchar)
        code = self.codeEncoder(inputcode, inputcodechar, ad)
        codeneg = self.codeEncoder(inputcodeneg, inputcodenegchar, adneg)
        nl = self.maxPoolnl(self.poolConvnl(nl.permute(0, 2, 1))).squeeze(-1)
        code = self.maxPoolcode(self.poolConvcode(code.permute(0, 2, 1))).squeeze(-1)
        codeneg = self.maxPoolcode(self.poolConvcode(codeneg.permute(0, 2, 1))).squeeze(-1)
        good_score = self.scoring(nl, code)
        bad_score = self.scoring(nl, codeneg)
        loss = (self.margin - good_score + bad_score).clamp(min=1e-6).mean()
        return loss, good_score, bad_score










