import plotsinwave
import matplotlib.pyplot as plt
import time
start = time.time()

number_nodes = 128
refractory_period = 500
spon_prob = 0.01
# first_node = 128
steps = 8000

lattice = plotsinwave.lattice(number_nodes, refractory_period, spon_prob, steps)

lattice.plotwave()
plt.show()

end = time.time()
print(end - start)