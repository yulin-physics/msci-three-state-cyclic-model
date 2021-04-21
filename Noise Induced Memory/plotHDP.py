import matplotlib.pyplot as plt
import numpy as np

with open('C:\\Users\PC\Desktop\MSci project\Hamming distance\HD.txt','r') as f:
    y = f.readlines()

hd = []

for i in range(len(y)):
    hd.append(eval(y[i]))

sum = [sum(x) for x in zip(*hd)]
print(len(y))
average = [x / len(y) for x in sum]
x= np.linspace(0.01,0.99,99)

plt.plot(x, average, '.')
plt.yscale('log')
plt.xlabel('p')
plt.ylabel('Hamming distance')
plt.show()