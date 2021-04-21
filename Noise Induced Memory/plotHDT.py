import matplotlib.pyplot as plt
import numpy as np

with open('HDtdiff.txt','r') as f:
    y = f.readlines()

y0 = eval(y[0])[2:]
y1 = eval(y[1])[2:]


l0 = len(y0)
l1 = len(y1)

a = y0

l = len(a)+1
t = list(range(1,l))

fig = plt.figure()
plt.plot(t[:100], a[:100], '.', label='periodic')
plt.title("Hamming distance as a function of time ")
# plt.plot(t, a, '.',label='periodic boundary condition')
# slope, intercept = np.polyfit(np.log(t)[0:100], np.log(a)[0:100], 1)
# print(slope)


plt.yscale('log')
plt.xscale('log')
plt.xlabel('Δt')
plt.ylabel('<D(Δt)>')
plt.legend()
plt.savefig('123.jpg', dpi = 1500)
# plt.title(slope)
plt.show()
