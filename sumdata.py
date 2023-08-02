import pickle
data1 = pickle.load(open('pjavadata.pkl', 'rb'))
data2 = pickle.load(open('ppythondata.pkl', 'rb'))
data3 = pickle.load(open('pcsharpdata.pkl', 'rb'))
data = data1 + data2 + data3
pickle.dump(data, open('pmixdata.pkl', 'wb'))