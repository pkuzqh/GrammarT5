import os
from tqdm import tqdm
import pickle
import json
import numpy as np
lst = ['train', "test1"]#["train", "dev", "test1"]
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
rules = tokenizer.get_vocab()
onelist = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers"]
rulelist = []
fatherlist = []
fathername = []
depthlist = []
copynode = {}
class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
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
def parseTree(treestr):
    tokens = treestr.split()
    root = Node(tokens[0], 0)
    currnode = root
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            if tokens[i + 2] == "^":
                x = x.lower()
            nnode = Node(x, i + 1)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
        else:
            currnode = currnode.father
    return root
def addException(root):
    if root.name in ['throws', 'types', 'label', 'case', 'goto']:
        for i, x in enumerate(root.child):
            nnode = Node('flag', 1)
            nnode.father = root
            nnode.child.append(x)
            x.father = nnode
            root.child[i] = nnode
    if root.name == 'value':
        if root.father.name == 'Assignment':
            root.name = 'value_assign'
    for x in root.child:
        addException(x)
maxnlnum = 40
hascopy = {}
def getcopyid(nls, name):
    global maxnlnum
    global hascopy
    lastcopyid = -1
    for i, x in enumerate(nls):
        if name.lower() == x.lower() and name != 'function' and name != 'int' and name != 'double' and name != 'boolean' and name != 'String':
            lastcopyid = i
            if i not in hascopy:
                hascopy[i] = 1
                return i + 1000000
    if lastcopyid != -1:
        return lastcopyid + 1000000
    return -1
rulead = np.zeros([641, 641])
astnode = {"pad": 0, "Unknown": 1}
nodess = {}
def getRule(node, nls, currId, d):
    global rules
    global onelist
    global rulelist
    global fatherlist
    global depthlist
    global copynode
    global rulead
    if node.name == "str_":
        assert(len(node.child) == 1)
    if len(node.child) == 0:
        return [], []
        if " -> End " not in rules:
            rules[" -> End "] = len(rules)
        return [rules[" -> End "]]
    child = node.child#sorted(node.child, key=lambda x:x.name)
    if len(node.child) == 1 and len(node.child[0].child) == 0 and ("identifier" in node.name or 'literal' in node.name):
        if len(node.child[0].child) != 0:
            print(node.child[0].name)
        if node.name == "type":
            print(node.printTree(node))
            
        if node.name == "string_literal":            
            if "string_literal -> srini_string" not in rules:
                rules["string_literal -> srini_string"] = len(rules)
            rulelist.append(rules["string_literal -> srini_string"])
            fatherlist.append(currId)
            fathername.append(node.name)
        else:
            nodess[node.name] = 1
            actions = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + node.child[0].name[:-4]))
            #for i in range(len(actions)):
            #    print(tokenizer.convert_ids_to_tokens(actions[i]), end=" ")
            #print(tokenizer.tokenize("sort the Array by the second element"))
            #assert(0)
            #actions.reverse()
            '''print(actions, tokenizer.tokenize(node.child[0].name[:-4]))
            if len(actions) > 512:
                actions = tokenizer.tokenize(node.child[0].name[:-4])#tokenizer.encode(node.child[0].name[:10])
                assert(0)'''
            for action in actions:
                rulelist.append(action)
                fatherlist.append(currId)
                fathername.append(node.name)
            rule = node.name + " -> End "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)
        currid = len(rulelist) - 1
    else:
        if node.name not in onelist:
            rule = node.name + " -> "
            for x in child:
                rule += x.name + " "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)
            currid = len(rulelist) - 1
            for x in child:
                getRule(x, nls, currid, d + 1)
        else:
            #assert(0)
            for x in (child):
                if (True):
                    rule = node.name + " -> " + x.name
                    if rule in rules:
                        rulelist.append(rules[rule])
                    else:
                        rules[rule] = len(rules)
                        rulelist.append(rules[rule])
                    fatherlist.append(currId)
                    fathername.append(node.name)
                getRule(x, nls, len(rulelist) - 1, d + 1)
            rule = node.name + " -> End "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)
    #return rulelist, fatherlistd
for j in range(0, 2):
    filename = lst[j]
    processdata = []
    #inputdir = x + "_input/"
    #outputdir = x + "_output/"
    data = pickle.load(open('%sdata.pkl'%filename, 'rb'))
    for i, entry in enumerate(tqdm(data)):
        nls = entry['input']
        root = entry['root']
        #print(root.printTree(root))
        #addException(root)
        rulelist = []
        fatherlist = []
        fathername = []
        try:
            getRule(root, nls, -1, 2)
        except:
            print(i)
            #assert(0)
            continue
        processdata.append({"rulelist": rulelist, "fatherlist": fatherlist, "fathername": fathername, 'input':nls})        
        rulelist = []
        fatherlist = []
        fathername = []

    #assert(0)
    open('processed%s.pkl'%filename, 'wb').write(pickle.dumps(processdata))
print(rules)
print(len(rules))
print(nodess)
open('rule2.pkl', 'wb').write(pickle.dumps(rules))
