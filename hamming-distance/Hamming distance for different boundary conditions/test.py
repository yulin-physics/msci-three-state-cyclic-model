import numpy as np
# exc_array = np.zeros(5, dtype=np.bool)
# print(exc_array)
# exc_array[1] = True
# print(exc_array)
#
# l = [1,2,3,4]
# for i in range(len(l)):
#     print(l[i])
#
#
# import numpy as np
# import  matplotlib.pyplot as plt
#
# s = (np.arange(256)/256)*2*np.pi
# x = np.arange(256)
# y = (64*np.sin(s)+64).astype("int")
# plt.plot(np.arange(256),(64*np.sin(s)+64).astype("int"))
# # X = [x for _,x in sorted(zip(y,x))]
# # y.sort()
# # print(X)
# # print(y)
# plt.show()
# for i in range(21):
#     indices = np.where(y == i)[0]
#     print(indices)
#
# print([0]*5)
#
# l = []
# initial_site = list(np.where(y == 0)[0])
# for i in range(len(initial_site)):
#     l.append(initial_site[i])
#
# print(l)
#
# t = list(range(10))
# print(t)


import matplotlib.pyplot as plt
import numpy as np

with open('HDtdiff.txt','r') as f:
    y = f.readlines()

y0 = eval(y[0])[2:]
y1 = eval(y[1])[2:]


l0 = len(y0)
l1 = len(y1)

a = y0
b = y1

k1 = len(a)+1
t1 = list(range(1,k1))

l2 = len(b)+1
t2 = list(range(1,l2))

print(l0,l1)

fig = plt.figure()
plt.plot(t1, a, '.',label='open boundary condition')
plt.title("Hamming distance as a function of time ")
plt.plot(t2, b, '.',label='periodic boundary condition')
# slope, intercept = np.polyfit(np.log(t)[0:100], np.log(a)[0:100], 1)
# print(slope)


plt.yscale('log')
plt.xscale('log')
plt.xlabel('Δt')
plt.ylabel('<D(Δt)>')
plt.legend()
plt.savefig('123.jpg', dpi = 600)
# plt.title(slope)
plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
#
# with open('HDtdiff.txt','r') as f:
#     y = f.readlines()
#
# y0 = eval(y[0])[2:]
# y1 = eval(y[1])[2:]
#
#
# l0 = len(y0)
# l1 = len(y1)
#
# a = y0
#
#
# k1 = len(a)+1
# t1 = list(range(1,k1))
#
#
#
# print(l0)
#
# fig = plt.figure()
# plt.plot(t1, a, '.',label='open boundary condition')
# plt.title("Hamming distance as a function of time ")
# # plt.plot(t2, b, '.',label='periodic boundary condition')
# # slope, intercept = np.polyfit(np.log(t)[0:100], np.log(a)[0:100], 1)
# # print(slope)
#
#
# plt.yscale('log')
# plt.xscale('log')
# plt.xlabel('Δt')
# plt.ylabel('<D(Δt)>')
# # plt.legend()
# # plt.savefig('123.jpg', dpi = 1500)
# # plt.title(slope)
# plt.show()
