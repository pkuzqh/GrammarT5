import subprocess
import time
import sys
lst = []
number = int(sys.argv[1])
dataset = sys.argv[2]
crad = [0, 1, 2, 3, 4, 5, 6, 7]
if dataset == 'mbpp':
    for i in range(number):
        p = subprocess.Popen(['CUDA_VISIBLE_DEVICES=%d python3 testmbpp.py %d %s'%(crad[i], i, dataset)], shell=True)
        time.sleep(1)
        lst.append(p)
    for x in lst:
        x.wait()
else:
    for i in range(number):
        p = subprocess.Popen(['CUDA_VISIBLE_DEVICES=%d python3 runold.py %d %s'%(crad[i], i, dataset)], shell=True)
        time.sleep(1)
        lst.append(p)
    for x in lst:
        x.wait()