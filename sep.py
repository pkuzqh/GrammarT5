import pickle
import random
import numpy as np
from tqdm import tqdm
import os
tmpdata = []#pickle.load(open('processdatapython.pkl', 'rb'))
tmpdata.extend(pickle.load(open('pmixdata.pkl', 'rb')))
newdata = []
for data in tmpdata:
    if 'nl' in data:
        newdata.append(data)
open('pmixdatawithnl.pkl', 'wb').write(pickle.dumps(newdata))