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
                child.append(Node(',', 0))
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
                child.append(Node(',', 0))
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
                child.append(Node(',', 0))
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
                child.append(Node(',', 0))
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
                child.append(Node(',', 0))
                child.append(x)
        child.append(node2)
        root.child = child
    for x in root.child:
        giveLam(x)
identifiers = ['identifier', 'type_identifier', 'null_literal', 'decimal_integer_literal', 'character_literal', 'decimal_floating_point_literal', 'hex_integer_literal', 'string_literal']
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
for i in range(0, 8):
    f = open('outval%d.txt'%i, 'r')
    lines = f.readlines()
    for j in tqdm(range(0, len(lines))):
        if j % 2 == 0:
            tree = parseTree(lines[j])
            giveLam(tree)
            mergeIdentifier(tree)
            code = stringfy(tree)
            outf.write(code)
            outf.write('\n')
