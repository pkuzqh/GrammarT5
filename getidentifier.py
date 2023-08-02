import pickle
ruledict = pickle.load(open("csharprule.pkl", "rb"))
lst1 = []
print(len(ruledict))
for x in ruledict:
    lst = x.split(" ")
    for y in lst[2:]:
        if 'identifier' in y:
            print(x)
            lst1.append(ruledict[x])
            break
print(lst1)