import javalang
import os
#from ast import nodes
from graphviz import Digraph
import json
import pickle
from tqdm import tqdm
nodes = ['Annotation', 'AnnotationDeclaration', 'AnnotationMethod', 'ArrayCreator', 
 		 'ArrayInitializer', 'ArraySelector', 'AssertStatement', 'Assignment', 
 		 'BasicType', 'BinaryOperation', 'BlockStatement', 'BreakStatement', 'Cast', 
 		 'CatchClause', 'CatchClauseParameter', 'ClassCreator', 'ClassDeclaration', 
 		 'ClassReference', 'CompilationUnit', 'ConstantDeclaration', 'ConstructorDeclaration', 
 		 'ContinueStatement', 'Creator', 'Declaration', 'Documented', 'DoStatement', 
 		 'ElementArrayValue', 'ElementValuePair', 'EnhancedForControl', 'EnumBody', 
 		 'EnumConstantDeclaration', 'EnumDeclaration', 'ExplicitConstructorInvocation', 
 		 'Expression', 'FieldDeclaration', 'ForControl', 'FormalParameter', 'ForStatement', 
 		 'IfStatement', 'Import', 'InferredFormalParameter', 'InnerClassCreator', 
 		 'InterfaceDeclaration', 'Invocation', 'LambdaExpression', 'Literal', 
 		 'LocalVariableDeclaration', 'Member', 'MemberReference', 'MethodDeclaration', 
 		 'MethodInvocation', 'MethodReference', 'PackageDeclaration', 'Primary', 'ReferenceType', 
 		 'ReturnStatement', 'Statement', 'StatementExpression', 'SuperConstructorInvocation', 
 		 'SuperMemberReference', 'SuperMethodInvocation', 'SwitchStatement', 'SwitchStatementCase', 
 		 'SynchronizedStatement', 'TernaryExpression', 'This', 'ThrowStatement', 'TryResource', 
 		 'TryStatement', 'Type', 'TypeArgument', 'TypeDeclaration', 'TypeParameter', 
 		 'VariableDeclaration', 'VariableDeclarator', 'VoidClassReference', 'WhileStatement',
 		 'int', 'double', 'float', 'boolean', 'long', 'short', 'byte', '(', ')']

nodeVect = {k: v for v, k in enumerate(nodes)}
def getroottree(tokens):
    root = Node("root", 0)
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
def getLiteral(vals):
    for v in vals:
        if isinstance(v, str):
            print(v)
            if not(type(num(v)).__name__.strip() in nodeVect):
                global malformed
                malformed = True
            return type(num(v)).__name__
class Node:
    def __init__(self, name, s):
        self.name = name
        self.id = s
        self.father = None
        self.child = []
        self.treestr = ""
    def printTree(self, r):
        s = r.name.lower() + " "#print(r.name)
        if len(r.child) == 0:
            s += "^ "
            return s
        #r.child = sorted(r.child, key=lambda x:x.name)
        for c in r.child:
            s += self.printTree(c)
        s += "^ "#print(r.name + "^")
        return s
    def getTreestr(self):
        if self.treestr == "":
            self.treestr = self.printTree(self)
            return self.treestr
        else:
            return self.treestr
        return self.treestr
def visitTree(node, g):
    g.node(name=node.name + str(node.id))
    if node.father:
        g.edge(node.father.name + str(node.father.id), node.name + str(node.id))
    node.child = node.child#sorted(node.child, key=lambda x:x.name)
    for x in node.child:
        visitTree(x, g)
def drawtree(treestr, p):
    g = Digraph('测试图片1')
    tokens = treestr.split()
    assert(len(tokens) == 2 * tokens.count('^'))
    print(tokens)
    root = Node("root", 0)
    currnode = root
    for i, x in enumerate(tokens[1:]):
        #print(x)
        if x != "^":
            nnode = Node(x, i + 1)
            nnode.father = currnode
            currnode.child.append(nnode)
            currnode = nnode
        else:
            currnode = currnode.father
    visitTree(root, g)
    g.render(filename=p,view=False)
#g.view()
def generateAST(tree):
    sub = []
    if not tree:
        return ['None']
    curr = type(tree).__name__
    #print(curr)
    if curr in nodes:
        if False:
            sub.append(str(getLiteral(tree.children)))
        else:
            sub.append(curr)
            try:
                for x in tree.attrs:
                    if x == "documentation":
                        continue
                    if not getattr(tree, x):
                        continue
                    '''if type(getattr(tree, x)).__name__ not in nodes:
                        print(type(getattr(tree, x)).__name__)
                        continue'''
                    sub.append(x)
                    node = getattr(tree, x)
                    if isinstance(node, list):
                        if len(node) == 0:
                            sub.append("empty")
                            sub.append("^")
                        else:
                            for ch in node:
                                if type(ch).__name__ not in nodes:
                                    continue
                                subtree = generateAST(ch)
                                sub.extend(subtree)
                    elif isinstance(node, javalang.tree.Node):
                        subtree = generateAST(node)
                        sub.extend(subtree)
                    elif not node:
                        continue
                    elif isinstance(node, str):
                        tmpStr = node
                        tmpStr = tmpStr.replace("\'", "").replace(" ", "").replace("-", "").replace(":", "")
                        #tmpStr = "<string>" if " " in x[1] else x[1].replace("\n", "").replace("\r", "")
                        if "\t" in tmpStr:
                            tmpStr = "<string>"
                        if len(tmpStr.split()) == 0:
                            tmpStr = "<empty>"
                        if tmpStr[-1] == "^":
                            tmpStr += "<>"
                        sub.append(tmpStr)
                        sub.append("^")
                    elif isinstance(node, set):
                        for ch in node:
                            if type(ch).__name__ not in nodes:
                                continue
                            subtree = generateAST(ch)
                            sub.extend(subtree)
                    elif isinstance(node, bool):
                        sub.append(str(node))
                        sub.append("^")
                    else:
                        print(type(node))
                        assert(0)
                    sub.append("^")
            except AttributeError:
                assert(0)
                pass
        sub.append('^')
        return sub
    else:
        print(curr)
    return sub
proj = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']
ids = [range(1, 27), list(range(1, 6)) + list(range(7, 41)), list(range(1, 63)) + list(range(64, 93)) + list(range(94, 177)), range(1, 19), range(25, 29), range(1, 48), range(1, 17), range(1, 19), range(1, 27), range(1, 113), range(1, 7), range(1, 94), range(1, 23), list(range(3, 66)) + [1], range(1, 107), range(1, 39), list(range(1, 21)) + list(range(22, 28))]
res = []
for i, p in enumerate(proj):
    for idx in tqdm(ids[i]):
        os.system('defects4j checkout -p %s -v %db -w buggy'%(p, idx))
        s = os.popen('defects4j export -p classes.modified -w buggy').readlines()
        if len(s) != 1:
            continue
        s = s[-1]
        dirs = os.popen('defects4j export -p dir.src.classes -w buggy').readlines()[-1]
        try:
            lines1 = open("buggy/%s/%s.java"%(dirs, s.replace('.', '/')), "r").read().strip()
        except:
            continue
        tokens = javalang.tokenizer.tokenize(lines1)
        parser = javalang.parser.Parser(tokens)
        try:
            tree = parser.parse()
                #print(tree)
                #wf = open("temp/process%d_1.txt" % i, "w")
            treeroot1 = getroottree(generateAST(tree))
                #drawtree(treestr, 'temp/' + str(i) + "_1")
               # wf.write(lines1 + "\n")
        except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
            print(lines1)
            exit(0)
            #continue
        os.system('defects4j checkout -p %s -v %df -w fixed'%(p, idx))	
        s = os.popen('defects4j export -p classes.modified -w fixed').readlines()[-1]
        lines2 = open("fixed/%s/%s.java"%(dirs, s.replace('.', '/')), "r").read().strip()
        tokens = javalang.tokenizer.tokenize(lines2)
        parser = javalang.parser.Parser(tokens)
        try:
            tree = parser.parse()
                #print(tree)
                #wf = open("temp/process%d_1.txt" % i, "w")
            treeroot2 = getroottree(generateAST(tree))
                #drawtree(treestr, 'temp/' + str(i) + "_1")
               # wf.write(lines1 + "\n")
        except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
            print(lines2)
            exit(0)
            #continue
        #print(treeroot1.child[-1].child[-1].child[3].name)
        methods1 = []
        for method in treeroot1.child[-1].child[-1].child:
            if method.name == 'body':
                for y in method.child:
                    if y.name == 'MethodDeclaration':
                        methods1.append(y)
        methods2 = []
        for method in treeroot2.child[-1].child[-1].child:
            if method.name == 'body':
                for y in method.child:
                    if y.name == 'MethodDeclaration':
                        methods2.append(y)
        if (len(methods1) != len(methods2)):
            continue
        for j in range(len(methods1)):
            if methods1[j].getTreestr() != methods2[j].getTreestr():
                res.append({'old':methods1[j].getTreestr(), 'new':methods2[j].getTreestr(), 'oldtree':methods1[j].getTreestr(), 'newtree':methods2[j].getTreestr()})
                #print(len(methods1[j].getTreestr().split()))
open("data.pkl", "wb").write(pickle.dumps(res, protocol=4))
