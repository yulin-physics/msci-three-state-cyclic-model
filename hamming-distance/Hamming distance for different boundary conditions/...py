import numpy as np
import  matplotlib.pyplot as plt

s = (np.arange(256)/256)*2*np.pi
x = np.arange(256)
y = (64*np.sin(s)+64).astype("int")
plt.plot(np.arange(256),(-32*np.sin(s)+32).astype("int"))
X = [x for _,x in sorted(zip(y,x))]
y.sort()
print(X)
print(y)
plt.show()
