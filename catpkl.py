import pickle
from tqdm import tqdm
def readpickle(filename, debug=False):
    data = []
    pbar = tqdm()
    with open(filename, 'rb') as f:
        while True:
            if len(data) % 100000 == 0:
                print("%d data read" % len(data))
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
            if not isinstance(data[-1], dict):
                data = data[:-1]
            if debug and len(data) > 1000:
                break
            pbar.update(1)
    pbar.close()
    return data
lst = readpickle('traindatawithnl5.pkl')
rules = pickle.load(open('processdata/csharprule.pkl', 'rb'))
rrdic = {}
for x in rules:
    rrdic[rules[x]] = x
for x in lst:
    #print(rrdic[x['rulelist'][1]])
    if rrdic[x['rulelist'][1]] != 'start -> python':
        continue
    for r in x['rulelist']:
        print(rrdic[r])
    assert(0)
