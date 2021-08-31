import numpy as np
import matplotlib.pyplot as plt

class lattice:
    def __init__(self, number_nodes, refractory_period, spon_prob, first_node, steps):
        self.number_nodes = number_nodes
        self.refractory_period = refractory_period
        self.p = spon_prob
        self.first_node = first_node
        self.steps = steps

    def plotwave(self):
        exc_array = np.zeros(self.number_nodes, dtype=np.bool)
        ref_array = np.ones(self.number_nodes, dtype=np.bool)
        exc_index = [np.array([-1])] * self.refractory_period  # List of arrays where each list contains cells excited at t%R
        indices = np.arange(self.number_nodes)
        initial_site = self.first_node
        exc_array[initial_site] = True
        ref_array[initial_site] = False
        exc_index[0] = np.array([initial_site])  # Fill excited index
        self.exc_pos = [initial_site]  # Store sites excited at each timestep
        self.times = [0]  # Store time site was excited
        t = 0  # Initial time

        for i in range(self.refractory_period):
            exc_right = (exc_index[
                             t % self.refractory_period] + 1) % self.number_nodes  # Sites to the right of previously excited
            exc_left = (exc_index[
                            t % self.refractory_period] - 1) % self.number_nodes  # Sites to the left of previously excited
            exc_array *= False
            exc_array[exc_right] = True
            exc_array[exc_left] = True
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

        for i in range(self.refractory_period, self.steps):

            prob_array = np.random.rand(self.number_nodes) < self.p  # Bool array with T where cell spontaneously excites
            exc_right = (exc_index[t % self.refractory_period] + 1) % self.number_nodes  # Sites to the right of previously excited
            exc_left = (exc_index[t % self.refractory_period] - 1) % self.number_nodes  # Sites to the left of previously excited

            exc_array *= False
            exc_array[exc_right] = True
            exc_array[exc_left] = True
            exc_array += prob_array  # Add spontaneous excitations
            exc_array *= ref_array  # Remove refractory cells
            exc_ind = indices[exc_array]  # Find excited indices
            ref_array[exc_ind] = 0  # Cells which exc become refractory for next time step

            t += 1
            if t >= self.refractory_period:
                ref_array[exc_index[t % self.refractory_period]] = 1  # Cells which excited R steps ago enter resting state

            exc_index[t % self.refractory_period] = exc_ind  # Replace excited cell index
            self.exc_pos += exc_ind.tolist()  # Store excited cells for plotting
            self.times += [t] * len(exc_ind)  # Store time cell was activated at

        # fig1 = plt.figure()
        # plt.scatter(self.times, self.exc_pos, s=0.5)

    def plotleader(self):
        self.leaders = [self.first_node]
        self.ltimes = [0]  # Store time site was leader

        for i in range(len(self.times) - 1):
            if self.times[i + 1] - self.times[i] > 1:
                self.leaders += [self.exc_pos[i + 1]]
                self.ltimes += [self.times[i + 1]]
                # if self.times[i + 2] - self.times[i + 1] == 0:
                #     self.leaders += [self.exc_pos[i + 2]]
                #     self.ltimes += [self.times[i + 2]]


    def HDvp(self): #calculate the hamming distance between consecutive waves
        ps= []
        ts = []
        pn = self.exc_pos
        tn = self.times
        ln = self.leaders
        ltn = self.ltimes
        for i in range(2):
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
        difference = []

        zip_object = zip(time_sorted[0], time_sorted[1])
        diff = []
        for list1_i, list2_i in zip_object:
            diff.append(list1_i - list2_i)
            diff = list(map(abs, diff))
        a = sum(diff) / len(diff)
        return a

    def HDvt(self):
        pn = self.exc_pos  # self.exc_pos
        tn = self.times  # self.times
        ln = self.leaders  # self.leaders
        ltn = self.ltimes  # self.ltimes

        positions = []
        times = []

        for i in range(len(ltn) - 1):
            ind = tn.index(ltn[i + 1])
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

        x = ltn[:100]
        y = ham_dis[:100]
        return y