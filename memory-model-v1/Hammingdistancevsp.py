import matplotlib.pyplot as plt
import time
import plotsinwave
start = time.time()

x = 1
while x < 50:
    p = 0.01
    Hamming_distance = []
    x_p = []
    while p < 1:
        lattice = plotsinwave.lattice(256, 256, p, 256*100)
        lattice.plotwave()
        a = lattice.HDvp()
        Hamming_distance += [a]
        x_p += [p]
        p+=0.01

    with open('HD.txt', 'r') as original:
        data1 = original.read()
    with open('HD.txt', 'w') as modified:
        modified.write(str(Hamming_distance) + '\n' + data1)

    x+=1

# plt.plot(x_p, Hamming_distance, '.')
# plt.yscale('log')
# plt.xlabel('p')
# plt.ylabel('Hamming distance')
# plt.show()