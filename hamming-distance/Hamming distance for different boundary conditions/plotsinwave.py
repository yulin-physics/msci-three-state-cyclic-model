import numpy as np
import  matplotlib.pyplot as plt
from statistics import mean

s = (np.arange(256)/256)*2*np.pi
x = np.arange(256)
y = (32*np.sin(s)+32).astype("int")

class lattice:
    def __init__(self, number_nodes, refractory_period, spon_prob, steps):
        self.number_nodes = number_nodes
        self.lastnode = number_nodes - 1
        self.refractory_period = refractory_period
        self.p = spon_prob
        # self.first_node = first_node
        self.steps = steps
    def plotwave(self):
        exc_array = np.zeros(self.number_nodes, dtype=np.bool)
        ref_array = np.ones(self.number_nodes, dtype=np.bool)
        exc_index = [np.array([-1])] * self.refractory_period  # List of arrays where each list contains cells excited at t%R
        indices = np.arange(self.number_nodes)
        initial_site = list(np.where(y == 0)[0])
        for i in range(len(initial_site)):
            exc_array[initial_site[i]] = True
            ref_array[initial_site[i]] = False
        exc_index[0] = np.array([initial_site])  # Fill excited index

        self.exc_pos = initial_site.copy()
        # self.exc_pos = initial_site # Store sites excited at each timestep
        self.times = [0]*len(initial_site)  # Store time site was excited
        t = 0  # Initial time

        compare = [[0], [0]]
        leaders = initial_site.copy()
        leadertimes = [0]*len(leaders)


        for i in range(1,65):
            sin_indices = np.where(y == i)[0]
            exc_array *= False
            for i in range(len(sin_indices)):
                exc_array[sin_indices[i]] = True
            exc_array *= ref_array  # Remove refractory cells
            exc_ind = indices[exc_array]  # Find excited indices

            ref_array[exc_ind] = 0  # Cells which exc become refractory for next time step

            t += 1
            if t >= self.refractory_period:
                ref_array[
                    exc_index[t % self.refractory_period]] = 1  # Cells which excited R steps ago enter resting state

            exc_index[t % self.refractory_period] = exc_ind  # Replace excited cell index
            self.exc_pos += exc_ind.tolist()  # Store excited cells for plotting
            self.times += [t] * len(exc_ind)  # Store time cell was activated at

        for i in range(65,self.steps):

            prob_array = np.random.rand(self.number_nodes) < self.p  # Bool array with T where cell spontaneously excites
            exc_right = (exc_index[t % self.refractory_period] + 1) % self.number_nodes  # Sites to the right of previously excited
            exc_left = (exc_index[t % self.refractory_period] - 1) % self.number_nodes  # Sites to the left of previously excited

            try:
                a = np.where(exc_right == 0)[0][0]
                exc_right = np.delete(exc_right, a)
            except IndexError:
                pass

            try:
                b = np.where(exc_left == self.lastnode)[0][0]
                exc_left = np.delete(exc_left, b)
            except IndexError:
                pass

            # try:
            #     c = np.where(exc_right == 129)[0][0]
            #     exc_right = np.delete(exc_right, c)
            # except IndexError:
            #     pass
            #
            # try:
            #     d = np.where(exc_left == 127)[0][0]
            #     exc_left = np.delete(exc_left, d)
            # except IndexError:
            #     pass

            exc_array *= False
            exc_array[exc_right] = True
            exc_array[exc_left] = True
            exc_array += prob_array  # Add spontaneous excitations
            exc_array *= ref_array  # Remove refractory cells
            exc_ind = indices[exc_array]  # Find excited indices

            ref_array[exc_ind] = 0  # Cells which exc become refractory for next time step

            t += 1
            compare += [exc_ind.tolist()]
            compare.pop(0)
            # print(compare, type(compare[0]), len(compare[0]), type(compare[1]), len(compare[1]))
            # print(compare[0] == [], compare[1] != [])
            if compare[0] == [] and compare[1] != []:
                leaders += compare[1]
                leadertimes += [t] * len(compare[1])

            if t >= self.refractory_period:
                ref_array[exc_index[t % self.refractory_period]] = 1  # Cells which excited R steps ago enter resting state

            exc_index[t % self.refractory_period] = exc_ind  # Replace excited cell index
            self.exc_pos += exc_ind.tolist()  # Store excited cells for plotting
            self.times += [t] * len(exc_ind)  # Store time cell was activated at

        # print(self.times)
        # print(self.exc_pos)
        # fig1 = plt.figure()
        plt.scatter(self.times, self.exc_pos, s=0.5)
        # print(len(leadertimes))
        # x  = leadertimes[0:20000]
        # y = leaders[0:20000]
        # fig2 = plt.figure()
        # plt.scatter(x,y, s=0.5)
        # print(leadertimes,len(leadertimes))
        # print(leaders,len(leaders))
        self.leaders = leaders.copy()
        # print(self.leaders)
        self.leadertimes = leadertimes.copy()
        # plt.scatter(leadertimes, leaders, s=0.5)

    def HDvt(self):
        pn = self.exc_pos  # self.exc_pos
        tn = self.times  # self.times
        ln = self.leaders  # self.leaders
        ltn = self.leadertimes # self.ltimes
        positions = []
        times = []

        for i in range(len(ltn) - 1):
            if ltn[i+1] - ltn[i] > 0:
                ind = tn.index(ltn[i + 1]) #index of the real time the leader appear
                positions += [pn[:ind]]
                pn = pn[ind:]
                time_singlewave = tn[:ind]
                time_singlewave = [x - ltn[i] for x in time_singlewave]
                times += [time_singlewave]
                tn = tn[ind:]

        time_sorted = []
        for i in range(len(times)):
            time_sorted += [list(list(zip(*(sorted(zip(positions[i], times[i])))))[1])]
        difference = []
        for i in range(len(time_sorted)):
            # time_sorted[i]
            zip_object = zip(time_sorted[0], time_sorted[i])
            diff = []
            for list1_i, list2_i in zip_object:
                diff.append(list1_i - list2_i)
                diff = list(map(abs, diff))
            difference += [diff]

        ham_dis = []
        for i in range(len(difference)):
            ham_dis.append(sum(difference[i]) / len(difference[i]))
        # fig5 = plt.figure()
        # plt.plot(ltn[:-1], ham_dis, '.')
        # print(len(ltn),len(ham_dis))
        # plt.yscale('log')
        # plt.xscale('log')
        # print(len(ham_dis))
        # x = ltn[1:101]
        # y = ham_dis[1:101]
        x = ltn
        y = ham_dis
        return y

    def HDvp(self): #calculate the hamming distance between consecutive waves
        ps = []
        ts = []
        pn = self.exc_pos
        tn = self.times
        ln = self.leaders
        ltn = self.leadertimes
        for i in range(len(ltn) - 1):
            if ltn[i+1] - ltn[i] > 0:
                ind = tn.index(ltn[i + 1])
                ps += [pn[:ind]]
                pn = pn[ind:]
                time_singlewave = tn[:ind]
                time_singlewave = [x - ltn[i] for x in time_singlewave]
                ts += [time_singlewave]
                tn = tn[ind:]

        time_sorted = []
        for i in range(len(ts)):
            time_sorted += [list(list(zip(*(sorted(zip(ps[i], ts[i])))))[1])]

        for i in range(len(time_sorted)-1):
            zip_object = zip(time_sorted[i], time_sorted[i+1])
            diff = []
            for list1_i, list2_i in zip_object:
                diff.append(list1_i - list2_i)
                diff = list(map(abs, diff))
        plt.plot(np.arange(len(diff)),diff,'.')
        a = sum(diff) / len(diff)
        # print(a)
        return a