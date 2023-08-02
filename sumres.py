ans = []
for i in range(6):
    strs = open("resdetail%d.txt"%i, "r").read()
    lst = eval(strs)
    ans.extend(lst)
open("restotal.txt", "w").write(str(ans))