import pickle
import random
import numpy as np
from tqdm import tqdm
import os
tmpdata = []#pickle.load(open('processdatapython.pkl', 'rb'))
tmpdata.extend(pickle.load(open('pmixdata.pkl', 'rb')))
idxs = np.random.permutation(range(len(tmpdata)))
traindata = []
for idx in idxs:
    traindata.append(tmpdata[idx])
#train_data = traindata
chunklen = len(traindata) // 6
for i in range(6):
    f = open('traindata%d.pkl'%i, 'wb')
    for j in tqdm(range(chunklen * i, chunklen * i + chunklen)):
        f.write(pickle.dumps(traindata[j]))
    f.close()