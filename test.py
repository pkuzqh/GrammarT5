import subprocess
fname = 'ans-126-49.txt'
p = subprocess.Popen(['python', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate()
print(err, out)
