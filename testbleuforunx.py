from evaluator.CodeBLEU import calc_code_bleu
from bleunl import calbleu
import sys
dataset = sys.argv[1]
if dataset in ['commentjava', 'commentpython']:
    preds = []
    f = open("out.txt", "r")
    for line in f:
        preds.append(str(len(preds)) + '\t' + line.strip())
    f.close()
    f = open("tmp.txt", "w")
    lines = open("processdata/ground%s.txt"%dataset, "r").readlines()
    for i in range(len(preds)):
        f.write(str(i) + '\t' + lines[i].strip() + '\n')
    f.close()
    bleu = calbleu('tmp.txt', preds)
else:
    import json
    f = open('groundd.txt', 'r')
    lst = f.readlines()
    f.close()
    wf = open('groundd1.txt', 'w')
    for x in lst:
        tmp = json.loads(x)
        wf.write(tmp['code'].strip().replace('\n', ' ') + '\n')
    wf.close()

    bleu = calc_code_bleu.get_codebleu("groundd1.txt", 'outd.txt', 'java', benchmark=dataset)
print(bleu)