import pickle
data = pickle.load(open('rawtest1data.pkl', 'rb'))
datalen = len(data)
import numpy as np
ids = range(datalen)
#np.random.permutation(range(datalen))
ans = open('ground.txt', 'r').readlines()
wf = open('test1.txt', 'w')
newdata = []
splielen = 320
for i in range(6 * splielen):
    newdata.append(data[ids[i]])
    wf.write(ans[ids[i]])
for i in range(0, 6):
    with open('rawtest1data%d.pkl'%i, 'wb') as f:
        print(len(newdata[i * splielen: (i + 1) * splielen]))
        f.write(pickle.dumps(newdata[i * splielen: (i + 1) * splielen]))