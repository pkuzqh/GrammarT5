f = open('processdata/groundrepairme.txt', 'r')
ground = f.readlines()
f.close()
f = open('out.txt', 'r')
out = f.readlines()
f.close()
coun = 0
acc = []
for i in range(len(ground)):
    lstg = ground[i].replace(' ', '').replace('"', '\'').replace('r\'', '\'').strip()
    lsto = out[i].replace(' ', '').replace('"', '\'').replace('r\'', '\'').strip()
    #print(lstg, lsto)
    if lstg == lsto:    
        acc.append(i)
        coun += 1
#print(coun/1805, coun)
print(coun)
lst = []
for i in range(8):
    tmp = open('resdetail' + str(i) + '.txt', 'r').read()
    lst.extend(eval(tmp))
for i, x in enumerate(lst):
    if x:
        if i not in acc:
            print(i, ground[i], out[i])
'''for i in range(len(out)):
    if i not in acc:
        print(i, ground[i], out[i])'''