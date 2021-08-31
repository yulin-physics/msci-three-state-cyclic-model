
# This code applies closed boundary condition on a 1D lattice, so the largest distance bewteen two sites is only half the lattice length.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import line_profiler 
import atexit

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats) #call print_stats at the end of script

L = 2048 #System size
p = 0.01 #Probability of spontaneous excitation/node activation prob
R = L #Refractory period/time step delay from R to Q
T = 10**5
class GHCA:
    def __init__(self,size=L, prob=p, refrac=R, waves=T):
        self.L=size
        self.R=refrac 
        self.p=prob 
        self.T=waves
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
  

        t=0 #initial real time
        count = 1
        gate1=False
        while count<self.T:
              
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
             
             #self.exc_dump += exc_ind.tolist() #Store excited cells for plotting
             #self.times += [t] * len(exc_ind) #Store time cell was activated at
            
         
             #False if excited site list is empty, True otherwise
             exc_bool=any(self.exc_array)
             
             gate2=gate1*exc_bool
             if gate2 == True:
                 if len(exc_ind)>1:
                     shift = [abs(number - self.pos_list[-1]) for number in exc_ind]
                     m, i = min((v,i) for i,v in enumerate(shift))
                     exc_ind = exc_ind[i]
                 self.pos_list.append(int(exc_ind))        
                 gate1=False
                 count += 1
             if exc_bool == False:
                 gate1=True

             
        self.time_list = range(1, self.T+1)
        
        np.savetxt('2048.csv', [v for v in zip(self.time_list, self.pos_list)],fmt='%i', delimiter=',')
         
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

    def plot_leaderDistribution(self):
        #distribution of leader drift in position in sebsequent wavefronts
        diff_x = []
        
        data = np.loadtxt('4096.csv', delimiter=',')
        #print(data[:,0])

        for i, pos in enumerate(data[1:,1]):
            diff = abs(pos - data[i-1,1])
            max_diff = self.L/2
            if diff > max_diff:
                diff = self.L - diff
            diff_x.append(int(diff))

        counts = dict()
        for i in diff_x:
            counts[i] = counts.get(i, 0) + 1
        #my_dict = {i:diff_x.count(i) for i in diff_x}
   
        diff_x = list(counts.keys())
        count = list(counts.values())
        
        df = pd.DataFrame({'dx': diff_x, 'height':count})
        df.to_csv('4096_dn.csv', index=False, header=None)
        
        plt.figure(2)
        ax = plt.gca()
        ax.plot(diff_x, count,'.', color='k')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Δn')
        ax.set_ylabel('P(Δn)')
        
    def powerfit(self, x, y, xnew):
        """line fitting on log-log scale"""
        k, m = np.polyfit(np.log(x), np.log(y), 1)
        return np.exp(m) * xnew**(k), k    
            

    def plot_leaderDrift(self, pos):
        #leader drift in position as a function of time lag
        n = []
        t = np.arange(1,10**4)
        
      
        for i in t:
            div = np.absolute(pos[i:] - pos[:-i])
            m = np.mean(div)
            n.append(m)
            
        
        # log-scaled bins
        edges = np.logspace(0, 4, 13)
        widths = abs(edges[1:] - edges[:-1])
        points = edges[:-1] + widths/2
        index = [int(round(x)-1) for x in edges]
        
        dn_list = []

        for j, val in enumerate(index[:-1]):
            ind_a = index[j]
            ind_b = index[j+1]
            dn = np.mean(n[ind_a: ind_b])
            dn_list.append(dn)


        plt.xscale('log')
        plt.yscale('log')
        plt.scatter(points, dn_list, s=20, color='skyblue')
       
        ys = self.powerfit(points[0:2], dn_list[0:2], points[0:3])
    
    
        plt.plot(points[0:3], ys[0], color='k')  
        plt.text(9,41,'H = %s'% float('%.2g' % ys[1]), fontsize=16)
        plt.text(9,40,'p = %s'% self.p, fontsize=16)
        #plt.text(6,650,'H = %s'% float('%.2g' % ys[1]), fontsize=16)
        print(ys[1])
        plt.xlabel('Δt')
        plt.ylabel('Δn')
        plt.show()
           
        
        
        
    


lattice = GHCA()
lattice.run()

