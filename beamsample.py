onelist_java = ['argument_list', 'formal_parameters', 'block', 'array_initializer', 'switch_block', 'type_arguments', "method_declaration", "modifiers", 'annotation_argument_list', 'variable_declarator', 'throws', 'element_value_array_initializer', 'annotation_argument_list', 'switch_block_statement_group', 'class_body', 'catch_type', 'assert_statement', 'try_statement', 'local_variable_declaration', 'try_statement', 'constructor_body', 'type_parameters', 'resource_specification', 'inferred_parameters', 'try_with_resources_statement', 'inits', 'updates', 'conditions']
identifiers_java = ['identifier', 'type_identifier', 'null_literal', 'decimal_integer_literal', 'character_literal', 'decimal_floating_point_literal', 'hex_integer_literal', 'string_literal']
for i in range(len(onelist_java)):
    onelist_java[i] += '_java'
for i in range(len(identifiers_java)):
    identifiers_java[i] += '_java'
onelist_py = ['argument_list', 'block', 'tuple', 'list', 'expression_list', 'subscript', 'conditional_expression', 'tuple', 'comparison_operator', 'body', 'lambda_parameters', 'dictionary', 'tuple', 'slice', 'set', 'assert_statement', 'if_statement', 'pattern_list', 'tuple_pattern', 'concatenated_string', 'import_from_statement', 'list_pattern', 'try_statement', 'parameters', 'expression_statement', 'print_statement', 'module', "dictionary_comprehension", "global_statement", "decorated_definition", "generator_expression", "dotted_name"]

identifiers_py = ['identifier', 'integer', 'float', 'string_literal']
for i in range(len(onelist_py)):
    onelist_py[i] += '_py'
for i in range(len(identifiers_py)):
    identifiers_py[i] += '_py'
onelist_cs = ['argument_list', 'switch_expression_arm', 'block', 'anonymous_object_creation_expression', 'initializer_expression', 'switch_section', 'local_function_statement', 'parameter_list', 'type_argument_list', 'switch_body', 'query_expression', 'argument', 'preprocessor_call', 'switch_expression','array_rank_specifier','prefix_unary_expression','while_statement','interpolation','join_clause', 'for_statement', 'attribute_list', 'type_parameter_list', 'attribute_argument_list', 'bracketed_argument_list','class_declaration', 'compilation_unit', 'declaration_list', 'if_statement', 'order_by_clause', 'parameter', 'postfix_unary_expression', 'property_pattern_clause', 'query_continuation', 'try_statement', 'tuple_expression', 'tuple_pattern', 'type_parameter_constraints_clause', 'variable_declaration', 'tuple_type', 'with_initializer_expression', 'from_clause', 'lock_statement', 'group_clause']
identifiers_cs = ['integer_literal', 'real_literal', 'identifier', 'null_literal', 'verbatim_string_literal', 'boolean_literal', 'string_literal', 'label_name', 'escape_sequence', 'character_literal']
for i in range(len(onelist_cs)):
    onelist_cs[i] = onelist_cs[i] + "_cs"
for i in range(len(identifiers_cs)):
    identifiers_cs[i] = identifiers_cs[i] + "_cs"
onelist = []
identifiers = []
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import numpy as np
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
class SearchNode:
    def __init__(self, ruledict, mode='gen', lang='java'):
        if mode == 'gen':
            if lang == 'java':
                self.state = [ruledict['start -> java']]
            elif lang == 'python':
                self.state = [ruledict['start -> python']]
            elif lang == 'csharp':
                self.state = [ruledict['start -> csharp']]        
            if lang == 'java':
                self.expandednode = ['java']
            elif lang == 'python':
                self.expandednode = ['python']
            elif lang == 'csharp':
                self.expandednode = ['csharp']
        if  mode == 'fill':
            self.state = [ruledict['<extra_id_0>']]
            self.expandednode = ['<extra_id_0>']
        if mode == 'nl':
            self.state = [ruledict['<s>']]
            self.expandednode = ['<s>']
        self.prob = 0
        self.mode = mode
        self.finished = False
    def checkapply(self, rule, rrdict):
        global onelist, identifiers
        rules = rrdict[rule]
        lst = rules.strip().split()
        expandedname = self.expandednode[-1]
        if "->" not in rules or lst[0] == '->':
            if lst[0] == '->' and expandedname != 'string_literal':
                return False
            else:
                if expandedname not in identifiers:
                    return False
        else:
            rules = rrdict[rule]
            if rules.strip().split()[0].lower() != expandedname.lower():
                return False
        return True
    def apply(self, rule, rrdict, prob, expanded):
        global onelist, identifiers
        rules = rrdict[rule]
        if self.mode == 'nl':
            if rules == '</s>':
                self.finished = True
                return
            self.expandednode.append(rules)
            self.prob = prob
            #print(self.state, rule)
            self.state.append(rule)
            return
        #print(rules)
        #lst = rules.strip().split()
        expandedname = self.expandednode[-1]  
        self.prob = prob          
        self.state.append(rule)            
        #print(rules)
        lst = rules.strip().split()
        if len(lst) <= 2:
            if 'Ġ' in rules and 'string_literal' not in expandedname:
                self.expandednode = self.expandednode[:-1]
        else:
            if rules.strip() == expandedname + " -> End":
                self.expandednode = self.expandednode[:-1]
            else:
                if expandedname not in onelist:
                    self.expandednode = self.expandednode[:-1]
                lst = rules.strip().split()[2:]
                for i in range(len(lst) -1, -1, -1):
                    if lst[i] in expanded:
                        self.expandednode.append(lst[i])
        if len(self.expandednode) == 0:
            self.finished = True 
class finishsetBm:
    def __init__(self, beamsize, length_penalty=0.1):
        self.beamsize = beamsize
        self.set = []
        self.length_penalty = length_penalty
        self.minprob = -1e10
        self.minidx = -1
    def add(self, node):
        score = node.prob / (len(node.state) ** self.length_penalty)
        if len(self.set) < self.beamsize:
            node.prob = score
            self.set.append(node)
            if self.minprob == -1e10:
                self.minprob = score
                self.minidx = 0
            elif score < self.minprob:
                self.minprob = score
                self.minidx = len(self.set) - 1
        else:
            if score > self.minprob:
                node.prob = score
                self.set[self.minidx] = node
                self.minprob = 1e10
                for i in range(len(self.set)):
                    score = self.set[i].prob
                    if score < self.minprob:
                        self.minprob = score
                        self.minidx = i
    def isfinish(self, prob, curlen):
        if len(self.set) < self.beamsize:
            return False
        else:
            if prob / (curlen ** self.length_penalty) > self.minprob:
                return False
            else:
                return True

    def finalize(self):
        self.set = sorted(self.set, key=lambda x:x.prob, reverse=True)
class BeamSearch:
    def __init__(self, beamsize, ruledict, length_penalty=0.1, temperature=1.0):
        self.beamsize = beamsize
        self.length_penalty = length_penalty
        self.expandedname = []
        self.valid = {}
        self.temperature = temperature
        idenid = []
        for x in ruledict:
            tmpname = x.strip().split()[0]
            if len(x.strip().split()) < 3:
                idenid.append(ruledict[x])
                continue
            self.expandedname.append(tmpname)
            self.valid.setdefault(tmpname, []).append(ruledict[x])
        self.expandedname.extend(identifiers_java + identifiers_py + identifiers_cs)
        for x in identifiers_java + identifiers_py + identifiers_cs:
            self.valid.setdefault(x, []).extend(idenid)
        hole = list(range(len(ruledict)))
        self.nlword = len(idenid)

        for i in range(100):
            self.valid.setdefault('<extra_id_' + str(i) + '>', []).extend(hole)
        for x in self.valid:
            self.valid[x] = sorted(list(set(self.valid[x])))
        self.rrdict = {}
        for x in ruledict:
            self.rrdict[ruledict[x]] = x
        self.ruledict = ruledict
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            print("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )


            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
    @torch.no_grad()
    def search(self, inputnl, model, mode='gen', lang='java', max_len=100, vocabsize=0):
        global onelist_java, onelist_py, identifiers_java, identifiers_py, onelist, identifiers, onelist_cs, identifiers_cs
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        if lang == 'java':
            onelist = onelist_java
            identifiers = identifiers_java
        elif lang == 'python':
            onelist = onelist_py
            identifiers = identifiers_py
        elif lang == 'csharp':
            onelist = onelist_cs
            identifiers = identifiers_cs
        batch_size = inputnl.size(0) // self.beamsize
        score = torch.zeros(batch_size, self.beamsize).to(inputnl.device)
        score.fill_(-1e20) 
        beams = {}
        finalbeams = {}
        past_key_values = None
        encodenl, nlmask = model.encode_nl(inputnl)
        for i in range(batch_size):
            beams[i] = [SearchNode(self.ruledict, mode=mode, lang=lang)]
            score[i, 0] = 0
            finalbeams[i] = finishsetBm(self.beamsize, self.length_penalty)
        codelen = max_len
        output_attention = None
        output_hiddenstates = None
        index = 0
        endnum = {}
        tmpstates = []
        for i in range(batch_size):
            tmpstates.append(beams[i][0].state)
            for j in range(self.beamsize - 1):
                tmpstates.append([0])
        while True:        
            tmpbeam = {}               
            if len(endnum) == batch_size:
                break
            if index == codelen:
                break
            tmpstates = torch.tensor(tmpstates).to(inputnl.device)
            output, pastkv = model.test_forward(encodenl, nlmask, tmpstates[:,-1:], past_key_values=past_key_values)
            validtensor = torch.zeros(batch_size, self.beamsize, vocabsize).to(inputnl.device)
            #print(self.rrdict[beams[0][0].state[-1]])
            if mode == 'nl':
                vid = torch.arange(self.nlword).to(inputnl.device)
                validtensor[:, :, vid] = 1
            else:
                for bh in range(batch_size):
                    if bh in endnum:
                        continue
                    for bm in range(self.beamsize):
                        if bm >= len(beams[bh]):
                            break
                        currindex = bh * self.beamsize + bm
                        validids = self.valid[beams[bh][bm].expandednode[-1]]
                        validtensor[bh, bm, validids] = 1
            validtensor = validtensor.reshape(batch_size * self.beamsize, -1)
            #print(torch.log(output[0, 0, 32141]))
            output = output.squeeze(1)
            output = torch.log(output)
            output = output.masked_fill(validtensor == 0, -1e10)
            #print(output.size())
            
            next_token_scores = output + score.reshape(batch_size * self.beamsize, 1).repeat(1, output.size(-1))
            #using temperature
            next_token_scores = next_token_scores / self.temperature
            #masknegative = torch.lt(next_token_scores, -5e10)

            #next_token_scores = next_token_scores.log_softmax(dim=-1)

            #next_token_scores = next_token_scores.masked_fill(masknegative, -1e20)
            #print(next_token_scores)
            #do sample            
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, self.beamsize * vocab_size)
            probs = F.softmax(next_token_scores, dim=-1)
            
            nexttokens = torch.multinomial(probs, num_samples=2 * self.beamsize)
            nexttokenscores = torch.gather(next_token_scores, -1, nexttokens)
            next_token_scores, _indices = torch.sort(nexttokenscores, descending=True, dim=-1)
            nexttokens = torch.gather(nexttokens, -1, _indices)
            
            nextindices = torch.div(nexttokens, vocab_size, rounding_mode='floor')
            nexttokens = nexttokens % vocab_size

            #print(nexttokenscores.size(), nextindices.size(), nexttokens.size())
            next_input_ids = []
            next_beam_id = []
            score.fill_(-1e20)
            for j in range(batch_size):
                maxscore = 0
                curlen = 0
                if j in endnum:
                    for i in range(self.beamsize):
                        next_input_ids.append([0] * (index + 2))
                        next_beam_id.append(0)
                    continue
                maxscore = nexttokenscores[j, 0].item()
                #print(maxscore)
                tmpbeams = []
                for k in range(2 * self.beamsize):
                    if len(tmpbeams) >= self.beamsize:
                        break

                    #if nexttokenscores[j, k] < -1e9:
                    #    break
                    ruleidx = nexttokens[j, k].item()
                    #if k == 0:
                    #    print(j, self.rrdict[ruleidx])
                    batchidx = nextindices[j, k].item()                    

                    originidx = j * self.beamsize + batchidx
                    #print(bh, bm, originidx, j, k, sortfinalscore[j, k].item())
                    #if(tmpstates.size(1) == 34):
                    #if batchidx == 0:
                    #    print(j, index, beams[j][batchidx].expandednode[-1], nexttokenscores[j, k].item(), nexttokens[j, k].item(), self.rrdict[nexttokens[j, k].item()])
                    orginbeam = beams[j][batchidx]
                    copynode = pickle.loads(pickle.dumps(orginbeam))
                    ruleidx = nexttokens[j, k].item()
                    copynode.apply(ruleidx, self.rrdict, nexttokenscores[j, k].item(), self.expandedname)
                    curlen = len(copynode.state)
                    if copynode.finished:
                        finalbeams[j].add(copynode)
                    else:
                        next_input_ids.append(copynode.state)
                        next_beam_id.append(originidx)
                        tmpbeams.append(copynode)
                        score[j, len(tmpbeams) - 1] = copynode.prob
                if len(tmpbeams) < self.beamsize:
                    for i in range(self.beamsize - len(tmpbeams)):
                        next_input_ids.append([0] * (index + 2))
                        next_beam_id.append(0)
                if finalbeams[j].isfinish(maxscore, curlen):
                    endnum[j] = 1
                beams[j] = tmpbeams
            past_key_values = self._reorder_cache(pastkv, torch.tensor(next_beam_id))
            tmpstates = next_input_ids
            index += 1
        for i in range(batch_size):
            if len(finalbeams[i].set) != 0:
                continue
            for j in range(self.beamsize):
                if j >= len(beams[i]):
                    break
                finalbeams[i].add(beams[i][j])
        for i in range(batch_size):
            finalbeams[i].finalize()
        return finalbeams
    def convertrulelist2tree(self, rulelist, lang='java', mode='gen'):
        if mode == 'nl':
            ans = []
            for i in range(1, len(rulelist)):
                ans.append(self.rrdict[rulelist[i]])
            if(len(ans) == 0):
                return "empty"
            return "".join(ans).replace("Ġ", " ")
        if mode == 'gen':
            if lang == 'java':
                root = Node('java', 1)
            elif lang == 'python':
                root = Node('python', 1)
            elif lang == 'csharp':
                root = Node('csharp', 1)
        else:
            root = Node('<extra_id_0>', 1)
        expanded = [root]
        for i in range(1, len(rulelist)):
            currexpanded = expanded[-1]
            rule = self.rrdict[rulelist[i]]
            lst = rule.strip().split()
            #print(currexpanded.name, rule)
            if len(lst) > 2:
                if rule.strip() == currexpanded.name + " -> End":
                    expanded = expanded[:-1]
                    if 'string_literal' in currexpanded.name:
                        currexpanded.child.reverse()
                    continue
                if currexpanded.name not in onelist:
                    #print(currexpanded.name)
                    expanded = expanded[:-1]
                if lst[0] != currexpanded.name:
                    print(lst[0], i, currexpanded.name)
                if currexpanded.name != '<extra_id_0>':
                    assert lst[0] == currexpanded.name
                for x in lst[2:]:
                    newnode = Node(x, 1)
                    currexpanded.child.append(newnode)
                for j in range(len(currexpanded.child) - 1, len(currexpanded.child) - len(lst[2:]) - 1, -1):
                    if currexpanded.child[j].name in self.expandedname:
                        expanded.append(currexpanded.child[j])
            else:
                newnode = Node(rule + '_ter', 1)
                currexpanded.child.append(newnode)
                if ('Ġ' in rule and 'string_literal' not in currexpanded.name):
                    expanded = expanded[:-1]
                    currexpanded.child.reverse()
        return root
        
