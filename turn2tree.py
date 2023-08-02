import ast
import javalang
import traceback
from tqdm import tqdm
from Searchnode1 import Node
import pickle
import sys
import json
sys.setrecursionlimit(1000000)
def readline(line):
    dic = eval(line)
    code = dic["code"]
    s_token = dic["docstring"]
    return code, s_token

def readfile(file_path):
    f = open(file_path)
    lines = f.readlines()
    f.close()
    code = []
    s_token = []
    for line in lines:
        c, s = readline(line)
        code.append(c)
        s_token.append(s)
    return code, s_token
def generateAST(tree):
    sub = []
    if not tree:
        return ['None', '🚀']
    if isinstance(tree, str):
        tmpStr = tree
        #tmpStr = tmpStr.replace(" ", "").replace(":", "")
        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
            tmpStr = tmpStr.replace("\t", " ").replace("\'", "").replace("\"", '')
        if len(tmpStr) == 0:
            tmpStr = "EmptyStr"
        sub.append(tmpStr)
        sub.append("🚀")
        return sub
    if isinstance(tree, list):
        if len(tree) == 0:
            sub.append("EmptyList")
            sub.append("🚀")
        else:
            for ch in tree:
                subtree = generateAST(ch)
                sub.extend(subtree)
        return sub
    curr = type(tree).__name__
    #print(curr)
    if True:
        if False:
            assert(0)#sub.append((str(getLiteral(tree.children)))
        else:
            sub.append(curr)
            try:
                for x in tree.attrs:
                    if x == "documentation":
                        continue
                    if x == "annotations":
                        continue
                    if not getattr(tree, x):
                        continue
                    '''if x == 'prefix_operators':
                        node = getattr(tree, x)
                        print(type(node))
                        print(len(node))
                        print(node[0])
                        assert(0)
                    if type(getattr(tree, x)).__name__ not in nodes:
                        print(type(getattr(tree, x)).__name__)
                        continue'''
                    sub.append(x)
                    node = getattr(tree, x)
                    if isinstance(node, list):
                        if len(node) == 0:
                            sub.append("EmptyList")
                            sub.append("🚀")
                        else:
                            for ch in node:
                                subtree = generateAST(ch)
                                sub.extend(subtree)
                    elif isinstance(node, javalang.tree.Node):
                        subtree = generateAST(node)
                        sub.extend(subtree)
                    elif not node:
                        continue
                    elif isinstance(node, str):
                        tmpStr = node
                        if "\t" in tmpStr or "'" in tmpStr or "\"" in tmpStr:
                            tmpStr = "srini_string"#tmpStr.replace("\t", " ").replace("\'", "").replace("\"", '')
                        if len(tmpStr) == 0:
                            tmpStr = "EmptyStr"
                        sub.append(tmpStr)
                        sub.append("🚀")
                    elif isinstance(node, set):
                        for ch in node:
                            subtree = generateAST(ch)
                            sub.extend(subtree)
                    elif isinstance(node, bool):
                        sub.append(str(node))
                        sub.append("🚀")
                    else:
                        print(type(node))
                        assert(0)
                    sub.append("🚀")
            except AttributeError:
                assert(0)
                pass
        sub.append('🚀')
        return sub
    else:
        print(curr)
    return sub
def readdata(lang, dataset, i):
    c = []
    s = []
    for i in range(i, i + 1):
        try:
            #print (f"jsonl/{dataset}/{lang}_{dataset}_{i}.jsonl")
            code, s_token = readfile(f"jsonl/{dataset}/{lang}_{dataset}_{i}.jsonl")
            c += code
            #print (code[0])
            #exit()
            s += s_token
        #    exit()
        except:
            break
            #print (str(e))
        #    break
    return c, s
def getroottree(tokens, isex=False):
    root = Node(tokens[0], 0)
    currnode = root
    idx1 = 1
    for j, x in enumerate(tokens[1:]):
        if x != "🚀":
            if tokens[j + 2] == '🚀':
                x = x + "_ter"
            nnode = Node(x, idx1)
            idx1 += 1
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
        else:
            currnode = currnode.father
    return root
def setProb(r, p):
  r.possibility =  p#max(min(np.random.normal(0.8, 0.1, 10)[0], 1), 0)
  #print(r.possibility)
  for x in r.child:
    setProb(x, p)
def getLocVar(node):
  varnames = []
  if node.name == 'VariableDeclarator':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'FormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  if node.name == 'InferredFormalParameter':
    currnode = -1
    for x in node.child:
      if x.name == 'name':
        currnode = x
        break
    varnames.append((currnode.child[0].name, node))
  for x in node.child:
    varnames.extend(getLocVar(x))
  return varnames
def doneParse (node, depth):
    if str(type(node)).replace("'", " ").replace(".", " ").split()[-2] == "list":
        print ("list ", end="")
        for i in range(len(node)):
            doneParse(node[i], depth + 1)
        print ("🚀 ", end="")
        return
    if str(type(node)).replace("'", " ").replace(".", " ").split()[-2] == "set":
        print ("set ", end="")
        for i in node:
            doneParse(i, depth  + 1)
        print ("🚀 ", end="")
        return
    if str(type(node)).replace("'", " ").replace(".", " ").split()[-2] == "str":
        print (node, end=" ")
        print ("🚀 ", end="")
        #for i in range(len(node)):
        #    doneParse(node[i])
        return
    if str(type(node)).replace("'", " ").replace(".", " ").split()[-2] == "bool":
        print ("bool " + str(node), end=" ")
        print ("🚀 ", end="")
        print ("🚀 ", end="")
        #for i in range(len(node)):
        #    doneParse(node[i])
        return
    if str(type(node)).replace("'", " ").replace(".", " ").split()[-2] == "Position":
    #    print ("bool " + str(node), end=" ")
        #for i in range(len(node)):
        #    doneParse(node[i])
        return
    if str(type(node)).replace("'", " ").replace(".", " ").split()[-2] == "NoneType":
    #    print ("bool " + str(node), end=" ")
        #for i in range(len(node)):
        #    doneParse(node[i])
        return
    
    print (str(type(node)).replace("'", " ").replace(".", " ").split()[-2] + " ", end="")
    for item in node.__dict__:
        if node.__dict__[item] == None or node.__dict__[item] == []:
            continue
        if str(item) == "_position":
            continue
        print (str(item) + " ", end="")
        doneParse(node.__dict__[item], depth + 1)
        print ("🚀 ", end="")
def gettype(t):
  if t in ['int_ter', 'double_ter', 'long_ter']:
    return 'numeric'
  elif t in ['boolean_ter']:
    return 'bool'
  elif 'String' in t:
    return 'string'
  else:
    return 'ptype'
def parserTree(filename):
    lines = (open(filename + '.json', 'r').readlines())
    data = []
    for i in tqdm(range(len(lines))):
        try:
            d = json.loads(lines[i])
            d['code'] = d['code'].replace("\"srini_string\" = \"srini_string\"", "\"srini_stirng\"")
            tokens = javalang.tokenizer.tokenize(d['code'])
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse_member_declaration()
            tmpc = generateAST(tree)
            root = getroottree(tmpc)
            data.append({'input':d['nl'], 'root':root})
            #print (str(tmpc) + " ")
            #assert(0)
                #print (str(node.__dict__))#.replace("=[]", " ").replace("=None", " ").replace("operator=", "operator ").replace("=[", " "))
                #break
                #print (type(node))
                #for k in node:
                #    print (k)
                #print (node.member)
                #print (node.name)
                #print (node.target)
                #print (node.)
                #print (node.children())
                #exit()
        except KeyboardInterrupt:
            exit(0)  
        except:
            traceback.print_exc()
            print(d['code'])
            #assert(0)
            pass
    open('%sdata.pkl'%filename, 'wb').write(pickle.dumps(data))
parserTree("train")
parserTree("test1")

