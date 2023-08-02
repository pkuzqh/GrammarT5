from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
from Searchnode1 import Node
from tqdm import tqdm
import json
import traceback
import pickle
JAVA_LANGUAGE = Language('parser/my-languages.so', 'java')
parser = Parser() 
parser.set_language(JAVA_LANGUAGE)
lines = []#f.read().splitlines()
import sys
sys.setrecursionlimit(1000000)
sonelist = ['formal_parameters', 'block', 'array_initializer', 'argument_list', 'type_arguments', 'annotation_argument_list']
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
def splitstr(s):
    lst = s.split()
    ans = []
    for x in lst:
        if x[0] == '(' and x[-1] == ')':
            ans += ['(', x[1:-1], ')']
        elif x[0] == '(' and x[-1] != ')':
            ans += ['(', x[1:]]
        elif x[0] != '(' and x[-1] == ')':
            ans += [x[:-1], ')']
        else:
            ans += [x]
    return ans
def revisitTree(root, newroot, s, cursor):
    global lines
    if root.type == 'comment':
        return
    newroot.name = root.type
    child = []
    #print(cursor.current_field_name(), dir(cursor))
    haschild = cursor.goto_first_child()
    for x in root.children:
        #if x.type in ['comment', ';', '(', ')', '[', ']', '{', '}', ',', '.']:
        #    cursor.goto_next_sibling()
        #    continue
        if root.type in sonelist and not x.is_named:
            cursor.goto_next_sibling()
            continue
        fname = cursor.current_field_name()
        tmp = None
        if fname is not None:#(newroot.name in ['binary_expression', 'ternary_expression', 'assignment_expression', 'enhanced_for_statement', 'for_statement', 'if_statement', 'class_body', 'switch_block_statement_group', 'local_variable_declaration', 'method_invocation', 'throws', 'array_access', 'assert_statement'] or 'expression' in newroot.name) and fname is not None:#fname is not None:
            tnode = Node(fname, 0)
            tmp = Node('init', 0)
            tnode.child.append(tmp)
            child.append(tnode)
            tmp.father = tnode
            tnode.father = newroot
        else:
            if x.type != 'comment':
                tmp = Node('init', 0)
                child.append(tmp)
                tmp.father = newroot
        if tmp is not None:
            revisitTree(x, tmp, s, cursor)
        cursor.goto_next_sibling()
    newroot.child = child
    if haschild:
        cursor.goto_parent()
    else:
        #print(dir(currsor))
        #print(dir(currsor.node.sexp()))
        s = cursor.node.start_point
        e = cursor.node.end_point
        
        ans = b''
        for idxx in range(s[0], e[0] + 1):
            line = lines[idxx].encode('utf-8')
            if idxx == s[0] and idxx != e[0]:
                ans += line[s[1]:]
            elif idxx == e[0] and idxx != s[0]:
                ans += line[:e[1]] 
            elif s[0] == e[0] and idxx == s[0]:
                ans += line[s[1]:e[1]]
            else:
                ans += line             
        #print(s, e, line[:s[1]], line[s[1]:e[1]], line[e[1]:], ans, root.type)
        ans = ans.decode('utf-8')
        #if root.type in ['identifier', 'predefined_type', 'integer_literal']:
        if root.type != ans:
            tnode = Node(ans, 0)
            newroot.child.append(tnode)
            tnode.father = newroot
        #assert(0)
wf = open('c1.txt', 'w')
def getMethod(root):
    ans = []
    if root.name == 'method_declaration' or root.name == 'constructor_declaration':
        ans.append(root)
    for x in root.child:
        ans.extend(getMethod(x))
    return ans
import pickle
def addter(root):
    if len(root.child) == 0:
        root.name += "_ter"
        return
    else:
        for x in root.child:
            print(root.name, x.name)
            addter(x)
        return
def simplifyFor(root):
    if root.name == 'for_statement':
        idx = []
        idloc = -1
        for i, x in enumerate(root.child):
            if x.name == ';_ter':
                idx.append(i)
            if x.name == 'init' and x.child[0].name == 'local_variable_declaration':
                idloc = i
        if len(idx) != 2:
            if len(idx) == 1 and idloc != -1:
                root.child.insert(idloc + 1, Node(';_ter', 0))
                idx[0] += 1
                idx.insert(0, idloc + 1)
            else:
                assert(0)
        inits = Node('inits', 0)
        conditions = Node('conditions', 0)
        updates = Node('updates', 0)
        inits.child = root.child[2:idx[0]]
        conditions.child = root.child[idx[0] + 1:idx[1]]
        updates.child = root.child[idx[1] + 1:-2]
        updates.father = root
        conditions.father = root
        inits.father = root
        root.child = root.child[:2] + [inits] + [root.child[idx[0]]] + [conditions] + [root.child[idx[1]]] + [updates] + root.child[-2:]
    for x in root.child:
        simplifyFor(x)
    return
def removeLam(root):
    if root.name == ';':
        root.father.child.remove(root)
        return
    for x in root.child:
        removeLam(x)
def parserTree(code):
        global lines
        try:
            #print(datas[i]['function'])
            line = "class A{" + code + "}"
            lines = line.splitlines()
            candidates = parser.parse(bytes(line,'utf8')).root_node
            cursor = candidates.walk()
            sroot = Node('init', 0)
            revisitTree(candidates, sroot, line, cursor)
            root = getMethod(sroot)
            addter(root[0])
            simplifyFor(root[0])
            removeLam(root[0])
            return root[0]
        except KeyboardInterrupt as e:
            assert(0)   
        except Exception as e:
            traceback.print_exc()