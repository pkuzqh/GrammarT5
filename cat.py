f = open('ground2.txt', 'r').readlines()
f2 = open('out.txt', 'r').readlines()
acc = []
for i in range(len(f)):
    if f[i].strip().split() == f2[i].strip().split():
        acc.append(i)
lst = []
for i in range(6):
    tmp = open('resdetail' + str(i) + '.txt', 'r').read()
    lst.extend(eval(tmp))
print(len(tmp), len(acc))
for i, x in enumerate(lst):
    if x:
        if i not in acc:
            print(i, f[i].split(), f2[i].split())
            print(f[i], f2[i])