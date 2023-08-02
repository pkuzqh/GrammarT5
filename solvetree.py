import os
from tqdm import tqdm
import pickle
import json
import numpy as np
import sys
lst = ['train', "test1"]#["train", "dev", "test1"]
from transformers import AutoTokenizer, AutoModel
import traceback
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
rules = pickle.load(open("pythonrule.pkl", "rb"))
onelist = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers", 'annotation_argument_list', 'variable_declarator', 'throws', 'element_value_array_initializer', 'annotation_argument_list', 'switch_block_statement_group', 'class_body', 'catch_type', 'assert_statement', 'try_statement', 'local_variable_declaration', 'try_statement', 'constructor_body', 'type_parameters', 'resource_specification', 'inferred_parameters', 'try_with_resources_statement', 'inits', 'updates', 'conditions']
identifiers = ['identifier', 'type_identifier', 'null_literal', 'decimal_integer_literal', 'character_literal', 'decimal_floating_point_literal', 'hex_integer_literal', 'string_literal']
sonelist = ['formal_parameters', 'block', 'array_initializer', 'argument_list', 'type_arguments', 'annotation_argument_list']
rulelist = []
fatherlist = []
fathername = []
depthlist = []
idenlist = []
expandnode = None
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
def getRule(node, currId, d):
    global rules
    global onelist
    global rulelist
    global fatherlist
    global idenlist
    global copynode
    global expandnode
    if node.name == "str_":
        assert(len(node.child) == 1)
    if len(node.child) == 0:
        return [], []
        if " -> End " not in rules:
            rules[" -> End "] = len(rules)
        return [rules[" -> End "]]
    if len(node.child) >= 1 and node.child[0].name == "identifier" and 'extra_id' in node.child[0].child[0].name:
        rulelist.append(rules['<' + node.child[0].child[0].name[:-4] + '>'])
        fatherlist.append(currId)
        idenlist.append(currId + 1)
        expandnode = node.name
        return
    if len(node.child) >= 2 and node.child[1].name == "identifier" and 'extra_id' in node.child[1].child[0].name:
        rulelist.append(rules['<' + node.child[1].child[0].name[:-4] + '>'])
        fatherlist.append(currId)
        idenlist.append(currId + 1)
        expandnode = node.name
        return
    child = node.child#sorted(node.child, key=lambda x:x.name)
    if len(node.child) == 1 and len(node.child[0].child) == 0 and ("identifier" in node.name or 'literal' in node.name):
        if len(node.child[0].child) != 0:
            print(node.child[0].name)
        if node.name == "type":
            print(node.printTree(node))
        if "extra_id" in node.child[0].name:            
            rulelist.append(rules['<' + node.child[0].name[:-4] + '>'])
            fatherlist.append(currId)
            fathername.append(node.name)
            idenlist.append(currId + 1)
            expandnode = 'identifier'
        else:
            nodess[node.name] = 1
            actions = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + node.child[0].name[:-4]))
            #for i in range(len(actions)):
            #    print(tokenizer.convert_ids_to_tokens(actions[i]), end=" ")
            #print(tokenizer.tokenize("sort the Array by the second element"))
            #assert(0)
            if node.name == "string_literal":
                #print(node.child[0].name)
                #print(tokenizer.tokenize(" " + node.child[0].name[:-4]))
                actions = actions[:25]
                node.child[0].name = "".join(tokenizer.convert_ids_to_tokens(actions)).replace("Ġ", " ").strip() + "_ter"
                actions = [rules["string_literal -> End"]] + actions
                #print(node.child[0].name)
                #assert(0)

            actions.reverse()
            '''print(actions, tokenizer.tokenize(node.child[0].name[:-4]))
            if len(actions) > 512:
                actions = tokenizer.tokenize(node.child[0].name[:-4])#tokenizer.encode(node.child[0].name[:10])
                assert(0)'''
            for action in actions:
                rulelist.append(action)
                fatherlist.append(currId)
                fathername.append(node.name)
            '''rule = node.name + " -> End "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)'''
        currid = len(rulelist) - 1
    else:
        if node.name not in onelist:
            rule = node.name + " -> "
            for x in child:
                rule += x.name + " "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                print(rule)
                assert(0)
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)
            currid = len(rulelist) - 1
            for x in child:
                getRule(x, currid, d + 1)
        else:
            for x in (child):
                if (True):
                    rule = node.name + " -> " + x.name
                    if rule in rules:
                        rulelist.append(rules[rule])
                    else:
                        print(rule)
                        assert(0)
                        rules[rule] = len(rules)
                        rulelist.append(rules[rule])
                    fatherlist.append(currId)
                    fathername.append(node.name)
                getRule(x, len(rulelist) - 1, d + 1)
            rule = node.name + " -> End "
            if rule in rules:
                rulelist.append(rules[rule])
            else:
                assert(0)
                rules[rule] = len(rules)
                rulelist.append(rules[rule])
            fatherlist.append(currId)
            fathername.append(node.name)
    #return rulelist, fatherlistd
def mergeIdentifier(root):
    if root.name in identifiers:
        if False:
            pass
        else:
            oname = ""
            for x in root.child:
                oname += x.name[:-4]
            oname += "_ter"
            nnode = Node(oname, 0)
            nnode.father = root
            root.child = [nnode]
    for x in root.child:
        mergeIdentifier(x)
    return
def parseTree(treestr):
    
    tokens = treestr.strip().split('🚀')[:-1]
    root = Node(tokens[0], 0)
    currnode = root
    for i, x in enumerate(tokens[1:]):
        if x != "^":
            nnode = Node(x, i + 1)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
        else:
            currnode = currnode.father
    return root
def filtererror(root):
    if root.name == 'ERROR' and len(root.child) != 0:
        return False
    for x in root.child:
        if not filtererror(x):
            return False
    return True
def getRuleList(root):
    global rulelist
    global fatherlist
    global idenlist
    global expandnode
    nroot = Node("java", 0)
    nroot.child = [root]
    root.father = nroot
    root = nroot
    rulelist = []
    fatherlist = []
    try:
        getRule(root, -1, 2)
    except:
        traceback.print_exc()
    rulelist = [rules['start -> java']] + rulelist
    fatherlist = [-1] + fatherlist
    fatherlist = [x + 1 for x in fatherlist]
    print(expandnode)
    return rulelist, fatherlist, idenlist, expandnode
if __name__ == '__main__':
    tmp = 0
    idx = int(sys.argv[1])
    data = pickle.load(open("datajava%d.pkl"%idx, "rb"))
    pdata = []
    lst1, lst2 = [], []
    for i, entry in enumerate(tqdm(data)):
        if i < 0:
            continue
        tree = entry['root']
        nl = entry['input']
        code = entry['code']
        root = parseTree(tree)
        nroot = Node("java", 0)
        nroot.child = [root]
        root.father = nroot
        root = nroot
        if not filtererror(root):
            continue
        rulelist = []
        fatherlist = []
        fathername = []
        try:
            getRule(root, -1, 2)
        except:
            traceback.print_exc()
            #assert(0)
            continue
        #mergeIdentifier(root)
        if "\"" in stringfy(root):
            code = stringfy(root)
        #print(len(tokenizer.encode(code)), len(rulelist))
            #assert(0)
        #code = stringfy(root)
        #if 'for_statement -> for_ter (_ter init condition ;_ter update )_ter body' in rules or 'for_statement -> for_ter (_ter init condition ;_ter update )_ter body ' in rules:
        #    print(code)
        #    print(tree.split('🚀')[:-1])
        #    assert(0)
        inputlist = [rules['start -> java']] + rulelist[:-1]
        lst1.append(len(tokenizer.encode(code)))
        lst2.append(len(inputlist))
        '''rrdict = {}
        for x in rules:
            rrdict[rules[x]] = x
        for i in range(len(inputlist)):
            print(rrdict[inputlist[i]])'''
        rulelist = [rules['start -> java']] + rulelist
        if len(rulelist) > 512:
            rulelist = []
            fatherlist = []
            fathername = []
            continue
        '''if nl != "":
            if tmp >= 10:
                assert(0)
            tmp += 1
            print(len(rulelist))
            print(nl)
            print(code)'''
        fatherlist = [-1] + fatherlist
        fatherlist = [x + 1 for x in fatherlist]
        if nl == "":
            pdata.append({'rulelist':rulelist, 'fatherlist':fatherlist})
        else:
            pdata.append({'nl':tokenizer.encode('<nl> ' + nl), 'rulelist':rulelist, 'fatherlist':fatherlist})
        rulelist = []
        fatherlist = []
        fathername = []
    print(np.mean(lst1), np.mean(lst2), rules['start -> java'])
    open('pjavadata%d.pkl'%idx, 'wb').write(pickle.dumps(pdata))
