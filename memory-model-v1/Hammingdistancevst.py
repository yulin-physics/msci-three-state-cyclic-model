import plotsinwave
import matplotlib.pyplot as plt
import numpy as np
import time
from statistics import mean
start = time.time()

number_nodes = 256
refractory_period = 256
spon_prob = 0.01
steps = 400*1000

listsofhd = []
n = 0
while n < 100:
    lattice = plotsinwave.lattice(number_nodes, refractory_period, spon_prob, steps)
    lattice.plotwave()
    listsofhd += [lattice.HDvt()]
    n += 1

# listsofhd = [ int(x) for x in listsofhd ]
unaveragedhd = [sum(x) for x in zip(*listsofhd)]
length = len(unaveragedhd)

Hamming_distance = [x / length for x in unaveragedhd]
t = list(range(1,101))

with open('HDtdiff.txt', 'r') as original:
    data1 = original.read()
with open('HDtdiff.txt', 'w') as modified:
    modified.write(str(Hamming_distance) + '\n' + data1)

fig = plt.figure()
plt.plot(t, Hamming_distance, '.')
slope, intercept = np.polyfit(np.log(t), np.log(Hamming_distance), 1)
print(slope)
plt.plot(t,10*(np.array(t)**0.2))
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Δt')
plt.ylabel('<D(Δt)>')
plt.title(slope)

plt.show()
end = time.time()
print(end - start)