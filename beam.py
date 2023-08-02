import torch
import pickle
from Dataset import pad_seq2
import numpy as np
onelist = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers", 'annotation_argument_list', 'variable_declarator', 'throws', 'element_value_array_initializer', 'annotation_argument_list', 'switch_block_statement_group', 'class_body', 'catch_type', 'assert_statement', 'try_statement', 'local_variable_declaration', 'try_statement', 'constructor_body', 'type_parameters', 'resource_specification', 'inferred_parameters', 'try_with_resources_statement', 'string_literal']
identifiers = ['identifier', 'type_identifier', 'null_literal', 'decimal_integer_literal', 'character_literal', 'decimal_floating_point_literal', 'hex_integer_literal']
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

class SearchNode:
    def __init__(self, ds, nl, name='', mode='gen', idenids=None):
        if mode == 'iden':
            self.currIdenIdx = 0
            self.idenids = idenids
            self.state = [ds.ruledict['<extra_id_0>']]
            self.parent = [self.idenids[self.currIdenIdx]]
            self.newidens = []
        else:
            self.state = [ds.ruledict["start -> java"]]
            self.parent = [0]
        self.prob = 0
        self.aprob = 0
        self.bprob = 0
        self.finish = False
        if mode == 'iden':
            self.root = Node(name, 0)
        else:
            self.root = Node("java", 2)
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
                if self.expanded.name == 'string_literal':
                    self.child.reverse()
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
            self.parent.append(self.idenids[self.currIdenIdx])
        else:
            self.parent.append(self.expanded.fatherlistID)
        assert(self.expanded.fatherlistID != -1)
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
    batch_size = (inputnl[0]).size(0)
    currNlLen = inputnl[0].shape[1]
    #print('------------------1')
    #print(inputnl[3][0])
    with torch.no_grad():
        beams = {}
        for i in range(batch_size):
            if mode == 'iden':
                beams[i] = [SearchNode(vds, [], name=name, mode=mode, idenids=inputnl[2])]
            else:
                beams[i] = [SearchNode(vds, [])]
        nlencode, nlmask = model.nl_encode((inputnl[0].cuda()), (inputnl[1].cuda()))
        index = 0
        endnum = {}
        continueSet = {}
        while True:
            print(index)
            currCodeLen = min(index + 2, 512)
            vds.Code_Len = min(index + 2, 512)#index + 1
            tmpbeam = {}
            ansV = {}
            if len(endnum) == batch_size:
                break
            #if index > 10:
            #    assert(0)
            if index >= currCodeLen:
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
                    print(x.printTree(x.root))
                    if not x.finish:
                        x.selectExpandedNode()
                    if x.expanded == None or len(x.state) >= currCodeLen:
                        x.finish = True
                        ansV.setdefault(i, []).append(x)
                    else:
                        validnum.append(i)
                        a = x.getRuleEmbedding(vds)
                        tmprule.append(a)
                        tmpAd.append(pad_seq2(x.parent, currCodeLen))
                #print("--------------------------")
                if len(tmprule) == 0:
                    continue
                batch_sizess = len(tmprule)
                tmprule = torch.tensor(tmprule)
                tmpAd = torch.tensor(tmpAd)
                bsize = batch_sizess
                if mode == 'iden': 
                    result = model.test_foward(nlencode[validnum], nlmask[validnum], (tmprule).cuda(), (tmpAd).cuda(), lefttree=True)
                else:
                    result = model.test_foward(nlencode[validnum], nlmask[validnum], (tmprule).cuda(), (tmpAd).cuda())
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
                        print(vds.rrdict[indexs[i].item()])
                        #print('-', indexs[i])
                        c = x.checkapply(indexs[i].item(), vds)
                        if not c:
                            tmpbeamsize += 1
                            continue
                        prob = x.prob + np.log(cresult[indexs[i]].item())#copynode.prob = copynode.prob + np.log(cresult[indexs[i]])
                        tmpbeam.setdefault(j, []).append([prob, indexs[i].item(), x])#tmpbeam.setdefault(j, []).append(copynode)
            for i in range(batch_size):
                if i in ansV:
                    if len(ansV[i]) == beamsize:
                        endnum[i] = 1
            for j in range(batch_size):
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