outf = open('out.txt', 'w')
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
def stringfy(node):
    ans = ""
    if len(node.child) == 0:
        if node.name[0] == 'Ġ':
            if 'string_literal' in node.father.name:
                ans += node.name[1:-4].replace('Ġ', ' ')
            else:
                ans += node.name[1:-4]
        else:
            ans = node.name[:-4]
    else:
        for x in node.child:
            ans += stringfy(x) + " "
    return ans
def parseTree(treestr):
    tokens = treestr.strip().split(' ')[:-1]
    #print(tokens)
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
def giveLam(root):
    if 'formal_parameters' in root.name:
        node1 = Node('(_ter', 0)
        node2 = Node(')_ter', 0)
        child = [node1]
        for i, x in enumerate(root.child):
            if i == 0:
                child.append(x)
            else:
                child.append(Node(',_ter', 0))
                child.append(x)
        child.append(node2)
        root.child = child
    if 'block' in root.name:
        node1 = Node('{_ter', 0)
        node2 = Node('}_ter', 0)
        child = [node1] + root.child + [node2]
        root.child = child
    if 'array_initializer' in root.name:
        node1 = Node('{_ter', 0)
        node2 = Node('}_ter', 0)
        child = [node1]
        for i, x in enumerate(root.child):
            if i == 0:
                child.append(x)
            else:
                child.append(Node(',_ter', 0))
                child.append(x)
        child.append(node2)
        root.child = child
    if 'argument_list' in root.name:
        node1 = Node('(_ter', 0)
        node2 = Node(')_ter', 0)
        child = [node1]
        for i, x in enumerate(root.child):
            if i == 0:
                child.append(x)
            else:
                child.append(Node(',_ter', 0))
                child.append(x)
        child.append(node2)
        root.child = child
    if 'type_arguments' in root.name:
        node1 = Node('<_ter', 0)
        node2 = Node('>_ter', 0)
        child = [node1]
        for i, x in enumerate(root.child):
            if i == 0:
                child.append(x)
            else:
                child.append(Node(',_ter', 0))
                child.append(x)
        child.append(node2)
        root.child = child
    if 'annotation_argument_list' in root.name:
        node1 = Node('(_ter', 0)
        node2 = Node(')_ter', 0)
        child = [node1]
        for i, x in enumerate(root.child):
            if i == 0:
                child.append(x)
            else:
                child.append(Node(',_ter', 0))
                child.append(x)
        child.append(node2)
        root.child = child
    for x in root.child:
        giveLam(x)
identifiers = ['identifier', 'type_identifier', 'null_literal', 'decimal_integer_literal', 'character_literal', 'decimal_floating_point_literal', 'hex_integer_literal', 'string_literal']
for i in range(len(identifiers)):
    identifiers[i] += '_java'
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
from tqdm import tqdm
import sys
tid = int(sys.argv[1])
lang = sys.argv[2]
dataset = sys.argv[3]
if dataset in ['repair', 'repairme']:
    from stringfy import strfy
else:
    from stringfy2 import strfy

for i in range(0, tid):
    f = open('outval%d.txt'%i, 'r')
    print(i)
    lines = f.readlines()
    if lang in ['nl']:
        for j in tqdm(range(0, len(lines))):
            outf.write(lines[j])
    else:
        for j in tqdm(range(0, len(lines))):
            if j % 2 == 0:
                if i == 7 and j / 2 == 991 - 126 * 7:
                    print(lines[j])
                    print(strfy(lines[j], lang))
                    #assert(0)
                code = strfy(lines[j], lang, neediden=dataset=='mbpp')
                if lang == 'java' and dataset not in ['repair', 'repairme']:
                    code = code.replace('>   >   >', '>>>').replace('>   >', '>>')
                if dataset in ['repair', 'repairme']:
                    code = code.replace('::', ': :').replace('->', '- >').replace('>>>', '> > >').replace('<<<', '< < <').replace('>>', '> >').replace('<<', '< <').replace(". class ", "class ")
                outf.write(code)
                outf.write('\n')
                #assert 0
