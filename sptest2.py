import pickle
import sys
datas = sys.argv[1]
data = pickle.load(open('processdata/%stest.pkl'%datas, 'rb'))
datalen = len(data)
datanum = int(sys.argv[2])
splielen = (datalen // datanum) + 1
for i in range(0, datanum):
    with open('fttest%d.pkl'%i, 'wb') as f:
        print(len(data[i * splielen: (i + 1) * splielen]))
        f.write(pickle.dumps(data[i * splielen: (i + 1) * splielen]))