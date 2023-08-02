import pickle
data = pickle.load(open('rawtraindata.pkl', 'rb'))
datalen = len(data)
splielen = datalen // 6
for i in range(0, 6):
    with open('rawtraindata%d.pkl'%i, 'wb') as f:
        f.write(pickle.dumps(data[i * splielen: (i + 1) * splielen]))