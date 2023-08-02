outf = open('out.txt', 'w')
simplestmt = ['assignment_py', 'return_statement_py', 'import_statement_py', 'raise_statement_py', 'pass_statement_py', 'delete_statement_py', 'yield_py', 'assert_statement_py', 'break_statement_py', 'continue_statement_py', 'global_statement_py', ' nonlocal_statement_py', 'import_from_statement_py', 'future_import_statement_py', 'expression_statement_py', 'exec_statement_py']
compound_stmt = ['if_statement_py', 'while_statement_py', 'for_statement_py', 'try_statement_py', 'with_statement_py', 'function_definition_py', 'class_definition_py', 'decorator_py']
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
def stringfy(node, hasspace=False):
    ans = ""
    if len(node.child) == 0:
        if node.name == 'updates':
            return ""
        if node.name[0] == 'Ġ':
            if 'string_literal' in node.father.name or 'character_literal' in node.father.name:
                ans += node.name[1:-4].replace('Ġ', ' ')
            else:
                ans += node.name[1:-4]
        else:
            ans = node.name[:-4]
    else:
        for k, x in enumerate(node.child):
            codestr = stringfy(x, hasspace)
            if 'method_invocation_java' in node.name:
                if ('java.' == ans[:5] or 'org.' == ans[:4]) and k == 2:
                    if 'METHOD' in codestr:
                        ans = ans + codestr + ' '
                    else:
                        ans = ans.replace(" ", "") + codestr + ' '
                else:
                    ans += codestr + ' ' 
            else:
                ans += codestr + ' '
    if 'scoped_type_identifier_java' in node.name:
        if "java" in ans or "android" in ans or 'org' in ans:
            ans = ans.replace(' ', '')
    if 'field_access_java' in node.name:
        if 'java' in ans or 'args   . length' in ans or 'org' in ans:
            ans = ans.replace(' ', '')
    if hasspace:
        if node.name == 'block_py':
            ans = ' <newline>' + ' <indent> ' + ans + ' <dedent> '
        if node.name in simplestmt:
            ans = ans + ' <newline> '
        if node.name in compound_stmt:
            ans = ans + ' <newline> '
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
    if 'for_statement' in root.name and 'enhance' not in root.name:
        if len(root.child[2].child) > 0 and root.child[2].child[-1].child[-1].name == 'local_variable_declaration_java':
            root.child = root.child[:3] + root.child[4:]
    elif 'formal_parameters' in root.name:
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
    elif 'block' in root.name and root.name[0] == 'b':
        node1 = Node('{_ter', 0)
        node2 = Node('}_ter', 0)
        child = [node1] + root.child + [node2]
        root.child = child
    elif 'array_initializer' in root.name:
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
    elif 'argument_list' in root.name:
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
    elif 'type_arguments' in root.name:
        if len(root.child) == 0 or 'type_arguments' not in root.child[0].name:
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
    elif 'annotation_argument_list' in root.name:
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
identifiers_java = ['identifier', 'type_identifier', 'null_literal', 'decimal_integer_literal', 'character_literal', 'decimal_floating_point_literal', 'hex_integer_literal', 'string_literal']
for i in range(len(identifiers_java)):
    identifiers_java[i] += '_java'
identifiers_py = ['identifier', 'integer', 'float', 'string_literal']
for i in range(len(identifiers_py)):
    identifiers_py[i] = identifiers_py[i] + '_py'
identifiers_cs = ['integer_literal', 'real_literal', 'identifier', 'null_literal', 'verbatim_string_literal', 'boolean_literal', 'string_literal', 'label_name', 'escape_sequence', 'character_literal']
for i in range(len(identifiers_cs)):
    identifiers_cs[i] = identifiers_cs[i] + "_cs"
identifiers = identifiers_java + identifiers_py + identifiers_cs

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
def strfy(treestr, lang, neediden=False):
    tree = parseTree(treestr)
    if lang == 'java':
        giveLam(tree)
    mergeIdentifier(tree)
    return stringfy(tree, lang=='python' and neediden)
import subprocess
def normalize(codestr, idx):
    lst = codestr.split()
    ans = ""
    currentBlock = ""
    for x in lst:
        if x == '<newline>':
            ans += '\n' + currentBlock
        elif x == '<indent>':
            currentBlock += '    '
            ans += '    '
        elif x == '<dedent>':
            currentBlock = currentBlock[:-4]
            ans += '\n' + currentBlock
        else:
            ans += x + ' '
    fname = 'tmp%s.py' % idx
    open(fname, 'w').write(ans)
    #print(ans)
    p = subprocess.Popen(['black', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    return