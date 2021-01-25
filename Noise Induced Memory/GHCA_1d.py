import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import binned_statistic
from collections import defaultdict
import line_profiler 
import atexit

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats) #call print_stats at the end of script

L = 1000 #System size
p = 0.01 #Probability of spontaneous excitation/node activation prob
R = 1000 #Refractory period/time step delay from R to Q
T = 100000 #Number of time steps to run for

class GHCA:
    def __init__(self,size=L, prob=p, refrac=R, tot_t=T):
        self.L=size
        self.R=refrac 
        self.p=prob 
        self.T=tot_t
        #self.initial_site = np.random.randint(L)
        self.initial_site = self.L//2
        self.my_dict = defaultdict(list) #dictionary of time:excited sites
        
        
    def initial_fire(self):
        
        self.exc_array = np.zeros(self.L,dtype=np.bool)
        self.ref_array = np.ones(self.L,dtype=np.bool)
        self.exc_index = [np.array([-1])] * self.R #List of arrays where each list contains cells excited at t%R
        
        self.exc_array[self.initial_site] = True
        self.ref_array[self.initial_site] = False
        self.exc_index[0] = np.array([self.initial_site]) #Fill excited index
        
        self.exc_dump = [self.initial_site] #Store sites excited at each timestep
        self.times = [0] #Store time site was excited
        self.my_dict[0] = self.initial_site
    
    @profile
    def run(self):
        initial=self.initial_fire()
        indices = np.arange(self.L)
        self.dict_array = [] #list of boolean values for the filtering of leader states
        
        t=0 #initial time
        for i in range(self.T):
              
             prob_array = np.random.rand(self.L)<self.p #gives a boolean array
             exc_right = (self.exc_index[t%self.R]+1)%self.L #Sites to the right of previously excited
             exc_left = (self.exc_index[t%self.R]-1)%self.L #Sites to the right of previously
             
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
             
             self.exc_dump += exc_ind.tolist() #Store excited cells for plotting
             self.times += [t] * len(exc_ind) #Store time cell was activated at
            
             self.my_dict[t] = exc_ind
             
             #False if list is empty, True otherwise
             self.dict_array.append(any(self.exc_array))
 
            
                
             #leader position trial
            # if len(self.exc_index[(t-1)%R])==0 and len(exc_ind)>=1:
             #    self.leader_time += [t] * len(exc_ind)
              #   self.leader_pos += exc_ind.tolist()
                 #but if waves overlap the leaders are not recorded
             
     
        #plt.figure(0)
        #plt.scatter(self.times,self.exc_dump,s=1, color='k')
        
        #plt.xlabel('Time(s)')
        #plt.ylabel('Space(n)')
        
        
    @profile
    def plot_leader(self):
        #consecutive positions of leaders        

        time_steps = np.arange(1, self.T+1)
        #pos_list = np.array(self.my_dict.values())
        inv_bool = np.invert(self.dict_array)
        inv_bool = np.concatenate(([False], inv_bool))

        bool_array = inv_bool[:-1] * self.dict_array
        

        time_temp = time_steps[bool_array]
        self.time_list = [0]    #time given in wavefront units
        self.pos_list = [self.initial_site] #list of positions of leaders
        time_true = [0]
        
        count = 1
        for keys in time_temp:
            pos = self.my_dict[keys]
            self.time_list += [count]*len(pos)
            self.pos_list += pos.tolist()
            time_true += [time_temp[count-1]] * len(pos)
            count += 1
        #pos_list = np.array(pos_list).flatten().tolist()

        plt.figure(1)
        plt.scatter(self.time_list, self.pos_list, s=0.8)
        plt.xlabel('Time')
        plt.ylabel('Leader(t)')
        #for i, j in zip(time_true, pos_list):
        #    plt.arrow(i-10, j,10,0, head_width=5, head_length=0.5, fc='k', ec='k')
        #for i in time_temp:
        #    plt.plot([i,i],[0,100],color='k')

    def plot_leaderDistribution(self):
        #distribution of leader drift in position in sebsequent wavefronts
        diff_x = []

        for i, pos in enumerate(self.pos_list[1:]):
            diff = abs(pos - self.pos_list[i-1])
            max_diff = self.L/2
            if diff > max_diff:
                diff = self.L - diff
            diff_x.append(diff)
        counts = dict()
        for i in diff_x:
            counts[i] = counts.get(i, 0) + 1
        #my_dict = {i:diff_x.count(i) for i in diff_x}
   
        diff_x = list(counts.keys())
        count = list(counts.values())
        
        plt.figure(2)
        ax = plt.gca()
        ax.plot(diff_x, count,'.')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Δn')
        ax.set_ylabel('P(Δn)')
        
        
        

    def plot_leaderDrift(self):
        #leader drift in position as a function of time lag
        diff_x = [] #drift in leader position
        diff_t = [] #time lag in units of wavefronts
        

        for i, pos_i in enumerate(self.pos_list):
            for j, pos_j in enumerate(self.pos_list[i+1:]):
                diff_xx = abs(pos_j - pos_i)
                diff_tt = abs(self.time_list[j] - self.time_list[i])
               
                max_diff = self.L/2
                if diff_xx > max_diff:
                    diff_xx = self.L - diff_xx
                  
                   
                diff_x.append(diff_xx)
                diff_t.append(diff_tt)
            

        
        plt.figure(3)
        ax = plt.gca()
        #ax.plot(diff_t, diff_x,'.')
        
        s, edges, _ = binned_statistic(diff_t,diff_x, statistic='mean', bins=np.logspace(0,4,12))

        #ax.hlines(s,edges[:-1],edges[1:], color="crimson", )


        ax.scatter(edges[:-1]+np.diff(edges)/2, s)
        
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Δt')
        ax.set_ylabel('Δn')

        
'''  
    def plot_leader(self):
        #leader drift in position as a function of time lag
        diff_x = []
        diff_t = [] #time lag in units of wavefronts

        for i, pos in enumerate(self.pos_list[1:]):
            diff_xx = abs(pos - self.pos_list[i-1])
            diff_tt = abs(self.times[i] - self.times[i-1])
            
            max_diff = self.L/2
            if diff_xx > max_diff:
                diff_xx = self.L - diff_xx
            diff_x.append(diff_xx)
            diff_t.append(diff_tt)

        
        # log-scaled bins: 0-1000 (excl.1000) and so on
        bins = np.logspace(0, 4, 10)
        #widths = (bins[1:] - bins[:-1])
        
        # Calculate histogram
        #counts, bin_edges = np.histogram(diff_t, bins=bins)
        
        # d is an index array holding the bin id for each point in diff_t
        #d = np.digitize(diff_t, bins)  
        
        stat, bin_edges, binnum = binned_statistic(diff_t, diff_t, statistic='max', bins=bins)
        
        x= []
        t= []
        for i in range(len(diff_t)):
            if diff_t[i] in stat:
                t.append(diff_t[i])
                x.append(diff_x[i])
                
            
        
        plt.figure(3)
        ax = plt.gca()
        ax.plot(t, x,'.')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Δt')
        ax.set_ylabel('Δn')
    
'''

lattice = GHCA()
lattice.run()
lattice.plot_leader()
#lattice.plot_leaderDistribution()
#lattice.plot_leaderDrift()

'''
i=0
while i<2:
    lattice = GHCA(size=L, refrac=R, tot_t=T)
    lattice.run()
    lattice.plot_leader()
    lattice.plot_leaderDrift()
    L=L*2
    R=R*2
    i+=1
'''    
