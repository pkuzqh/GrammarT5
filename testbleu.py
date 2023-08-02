from evaluator.CodeBLEU import calc_code_bleu
from bleunl import calbleu
import sys
dataset = sys.argv[1]
#print(calc_code_bleu.get_codebleu("ground%s.txt"%dataset, 'repairme.txt', 'java', benchmark=dataset))
#assert(0)
if dataset in ['commentjava', 'commentpython']:
    preds = []
    f = open("out.txt", "r")
    for line in f:
        preds.append(str(len(preds)) + '🚀' + line.strip())
        #preds.append(line.strip())
    f.close()
    f = open("tmp.txt", "w")
    lines = open("processdata/ground%s.txt"%dataset, "r").readlines()
    for i in range(len(preds)):
        f.write(str(i) + '🚀' + lines[i].strip() + '\n')
    f.close()
    bleu = calbleu('tmp.txt', preds)
else:
    if dataset == 'transj2c':
        #bleu = calc_code_bleu.get_codebleu("processdata/ground%s.txt"%dataset, 'out.txt', 'c_sharp', benchmark='transj2c')
        bleu = calc_code_bleu.get_codebleu("/data3/zqh/GrammarT5-base/processdata/data/transj2c/test.cs", 'codetrans.txt', 'c_sharp', benchmark='transj2c')
    else:
        #lines = open("processdata/ground%s.txt"%dataset, 'r').readlines()
        #lines2 = open("/data3/zqh/GrammarT5-base/processdata/data/repairme/medium/test.buggy-fixed.fixed").readlines()
        #for i in range(len(lines)):
        #    tmp = lines[i].replace(". class ", "class ")
        #    if tmp.replace(" ", "") != lines2[i].replace(" ", ""):
        #        print(tmp)
        #        print(lines2[i])
        bleu = calc_code_bleu.get_codebleu("processdata/ground%s.txt"%dataset, 'out.txt', 'java', benchmark=dataset)
        #bleu = calc_code_bleu.get_codebleu("/data3/zqh/GrammarT5-base/processdata/data/repairme/medium/test.buggy-fixed.fixed", 'out.txt', 'java', benchmark=dataset)

print(bleu)
