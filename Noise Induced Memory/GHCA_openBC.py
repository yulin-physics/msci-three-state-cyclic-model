# This code applies open boundary condition on a 1D lattice, ends of the lattice are not joined.

import numpy as np
import copy
import line_profiler 
import atexit

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats) #call print_stats at the end of script

L = 256 #System size
p = 0.01 #Probability of spontaneous excitation/node activation prob
R = L #Refractory period/time step delay from R to Q
T = 100000 #Number of wavefronts
perturb_p = 0.08
perturb_t = 100 #real time
class GHCA:
    def __init__(self,size=L, prob=p, refrac=R, waves=T, prob2=perturb_p, t2=perturb_t):
        self.L=size
        self.R=refrac 
        self.p=prob 
        self.T=waves
        self.t2=t2
        self.p2=prob2
        #self.initial_site = self.L//2
       
        
    def initial_fire(self):
        self.initial_site = np.random.randint(self.L)
        self.exc_array = np.zeros(self.L,dtype=np.bool)
        self.ref_array = np.ones(self.L,dtype=np.bool)
        self.exc_index = [np.array([-1])] * self.R #List of arrays where each list contains cells excited at t%R
        
        self.exc_array[self.initial_site] = True
        self.ref_array[self.initial_site] = False
        self.exc_index[0] = np.array([self.initial_site]) #Fill excited index
        
        #self.exc_dump = [self.initial_site] #Store sites excited at each timestep
        self.times = [0] #Store time site was excited
      
    
    @profile
    def run(self):
    #plots consecutive positions of leader for each wave
        
        self.initial_fire()
        self.time_list = [0]    #time given in wavefront units
        self.pos_list = [self.initial_site] #list of positions of leaders
        
        indices = np.arange(self.L)
        
# =============================================================================
#         #time varying probability
#         x = np.linspace(0.0, np.pi, num=self.T, endpoint=True)
#         cos_values = abs(np.cos(x))
#         prob_values = self.p*cos_values
#         c=0
# =============================================================================
        
        #t_before=0
        t=0 #initial real time
        count = 1
        gate1=False
        while count<self.T:
                
            prob_array = np.random.rand(self.L)<self.p
            #prob_array = np.random.rand(self.L)<prob_values[c] #gives a boolean array
            exc_right = (self.exc_index[t%self.R]+1)%self.L #Sites to the right of previously excited
            exc_left = (self.exc_index[t%self.R]-1)%self.L #Sites to the right of previously excited
             
            try:
                a = np.where(exc_right == 0)[0][0]
                exc_right = np.delete(exc_right, a)
            except IndexError:
                pass
            try:
                b = np.where(exc_left == L-1)[0][0]
                exc_left = np.delete(exc_left, b)
            except IndexError:
                pass
                 
            self.exc_array *= False
            self.exc_array[exc_right] = True
            self.exc_array[exc_left] = True
            self.exc_array += prob_array #Add spontaneous excitations
            self.exc_array *= self.ref_array #Remove refractory cells
            exc_ind = indices[self.exc_array] #indices of the excited state
             
            self.ref_array[exc_ind] = 0 
            t += 1

            if t >= self.R:
                self.ref_array[self.exc_index[t % self.R]] = 1 #Cells which excited R steps ago enter resting state
            self.exc_index[t % self.R] = exc_ind #Replace excited cell index
             
            #self.exc_dump += exc_ind.tolist() #Store excited cells for plotting
            #self.times += [t] * len(exc_ind) #Store time cell was activated at
            
            #False if excited site list is empty, True otherwise
            exc_bool=any(self.exc_array)
             
            #only stores leader positions
            gate2=gate1*exc_bool
            if gate2 == True:
                if len(exc_ind)>1:
                    shift = [abs(number - self.pos_list[-1]) for number in exc_ind]
                    m, i = min((v,i) for i,v in enumerate(shift))
                    #n, j = min((v,i) for i,v in enumerate(shift))
                    #temp = [i, j]
                    #new_i=np.random.randint(len(temp))
                    exc_ind = exc_ind[i]
                self.pos_list.append(int(exc_ind))        
                gate1=False
                count += 1
                #c += 1
                #print(t-t_before)
                #t_before = t
            if exc_bool == False:
                gate1=True
        
        self.time_list = range(1, self.T+1)
        
# =============================================================================
#         plt.scatter(self.time_list, self.pos_list, s=0.8, color='k')
#         plt.xlabel('Time')
#         plt.ylabel('Leader')
# =============================================================================

        np.savetxt('2048_open.csv', [v for v in zip(self.time_list, self.pos_list)],fmt='%i', delimiter=',')

# =============================================================================
#         plt.figure(1)
#         plt.scatter(self.time_list, self.pos_list, s=0.8, color='k')
#         plt.xlabel('Time')
#         plt.ylabel('Leader(t)')
# =============================================================================
        #for i, j in zip(time_true, pos_list):
        #    plt.arrow(i-10, j,10,0, head_width=5, head_length=0.5, fc='k', ec='k')
        #for i in time_temp:
        #    plt.plot([i,i],[0,100],color='k')

    def perturb(self):
        
        self.initial_fire()
        self.time_list = [0]    #time given in wavefront units
        self.pos_list = [self.initial_site] #list of positions of leaders
        
        indices = np.arange(self.L)
        
# =============================================================================
#         #time varying probability
#         x = np.linspace(0.0, np.pi, num=self.T, endpoint=True)
#         cos_values = abs(np.cos(x))
#         prob_values = self.p*cos_values
#         c=0
# =============================================================================
        
        #t_before=0
        t=0 #initial real time
        count = 1
        gate1=False
        while count<self.T:
            if t == self.t2:
                count_temp = count
                t_temp = t
                gate1_temp = gate1
                self.pos_perturb = copy.deepcopy(self.pos_list)
                self.exc_index_temp = copy.deepcopy(self.exc_index)
                self.ref_array_temp = copy.deepcopy(self.ref_array)
                ph = self.p2
                while count_temp<self.T:
                    
                    prob_array = np.random.rand(self.L)<ph
                    ph=self.p
                    exc_right = (self.exc_index_temp[t_temp%self.R]+1)%self.L #Sites to the right of previously excited
                    exc_left = (self.exc_index_temp[t_temp%self.R]-1)%self.L #Sites to the right of previously excited
                     
                    try:
                        a = np.where(exc_right == 0)[0][0]
                        exc_right = np.delete(exc_right, a)
                    except IndexError:
                        pass
                    try:
                        b = np.where(exc_left == L-1)[0][0]
                        exc_left = np.delete(exc_left, b)
                    except IndexError:
                        pass
                         
                    self.exc_array *= False
                    self.exc_array[exc_right] = True
                    self.exc_array[exc_left] = True
                    self.exc_array += prob_array #Add spontaneous excitations
                    self.exc_array *= self.ref_array_temp #Remove refractory cells
                    exc_ind = indices[self.exc_array] #indices of the excited state
                     
                    self.ref_array_temp[exc_ind] = 0 
                    t_temp += 1
        
                    if t_temp >= self.R:
                        self.ref_array_temp[self.exc_index_temp[t_temp % self.R]] = 1 #Cells which excited R steps ago enter resting state
                    self.exc_index_temp[t_temp % self.R] = exc_ind #Replace excited cell index
                     
                    #self.exc_dump += exc_ind.tolist() #Store excited cells for plotting
                    #self.times += [t] * len(exc_ind) #Store time cell was activated at
                    
                    #False if excited site list is empty, True otherwise
                    exc_bool=any(self.exc_array)
                     
                    #only stores leader positions
                    gate2=gate1_temp*exc_bool
                    if gate2 == True:
                        if len(exc_ind)>1:
                            shift = [abs(number - self.pos_perturb[-1]) for number in exc_ind]
                            m, i = min((v,i) for i,v in enumerate(shift))
                            #n, j = min((v,i) for i,v in enumerate(shift))
                            #temp = [i, j]
                            #new_i=np.random.randint(len(temp))
                            exc_ind = exc_ind[i]
                        self.pos_perturb.append(int(exc_ind))        
                        gate1_temp=False
                        count_temp += 1
                        #c += 1
                        #print(t-t_before)
                        #t_before = t
                    if exc_bool == False:
                        gate1_temp=True
            
           
                
            prob_array = np.random.rand(self.L)<self.p
            #prob_array = np.random.rand(self.L)<prob_values[c] #gives a boolean array
            exc_right = (self.exc_index[t%self.R]+1)%self.L #Sites to the right of previously excited
            exc_left = (self.exc_index[t%self.R]-1)%self.L #Sites to the right of previously excited
             
            try:
                a = np.where(exc_right == 0)[0][0]
                exc_right = np.delete(exc_right, a)
            except IndexError:
                pass
            try:
                b = np.where(exc_left == L-1)[0][0]
                exc_left = np.delete(exc_left, b)
            except IndexError:
                pass
                 
            self.exc_array *= False
            self.exc_array[exc_right] = True
            self.exc_array[exc_left] = True
            self.exc_array += prob_array #Add spontaneous excitations
            self.exc_array *= self.ref_array #Remove refractory cells
            exc_ind = indices[self.exc_array] #indices of the excited state
             
            self.ref_array[exc_ind] = 0 
            t += 1
        
            if t >= self.R:
                self.ref_array[self.exc_index[t % self.R]] = 1 #Cells which excited R steps ago enter resting state
            self.exc_index[t % self.R] = exc_ind #Replace excited cell index
             
            #self.exc_dump += exc_ind.tolist() #Store excited cells for plotting
            #self.times += [t] * len(exc_ind) #Store time cell was activated at
            
            #False if excited site list is empty, True otherwise
            exc_bool=any(self.exc_array)
             
            #only stores leader positions
            gate2=gate1*exc_bool
            if gate2 == True:
                if len(exc_ind)>1:
                    shift = [abs(number - self.pos_list[-1]) for number in exc_ind]
                    m, i = min((v,i) for i,v in enumerate(shift))
                    #n, j = min((v,i) for i,v in enumerate(shift))
                    #temp = [i, j]
                    #new_i=np.random.randint(len(temp))
                    exc_ind = exc_ind[i]
                self.pos_list.append(int(exc_ind))        
                gate1=False
                count += 1
                #c += 1
                #print(t-t_before)
                #t_before = t
            if exc_bool == False:
                gate1=True
        
        self.time_list = range(1, self.T+1)

# =============================================================================
#         plt.scatter(self.time_list, self.pos_list, s=0.8, color='k')
#         plt.xlabel('Time')
#         plt.ylabel('Leader')
# =============================================================================

        np.savetxt('256_perturb08_retry.csv', [v for v in zip(self.time_list, self.pos_list, self.pos_perturb)],fmt='%i', delimiter=',')
    
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


