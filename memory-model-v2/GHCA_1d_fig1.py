import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import binned_statistic
from collections import defaultdict
import line_profiler 
import atexit

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats) #call print_stats at the end of script

L = 128 #System size
p = 0.01 #Probability of spontaneous excitation/node activation prob
R = 128 #Refractory period/time step delay from R to Q
T = 5000 #Number of time steps to run for

class GHCA:
    def __init__(self,size=L, prob=p, refrac=R, tot_t=T):
        self.L=size
        self.R=refrac 
        self.p=prob 
        self.T=tot_t
        self.initial_site = np.random.randint(L)
        #self.initial_site = self.L//2
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
        self.initial_fire()
        indices = np.arange(self.L)
        self.pos_list=[self.initial_site]
        self.time_list=[0]
        
        gate1=False
        t=0 #initial time
        for i in range(self.T):
              
             prob_array = np.random.rand(self.L)<self.p #gives a boolean array
             exc_right = (self.exc_index[t%self.R]+1)%self.L #Sites to the right of previously excited
             exc_left = (self.exc_index[t%self.R]-1)%self.L #Sites to the right of previously
             
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
             
             self.exc_dump += exc_ind.tolist() #Store excited cells for plotting
             self.times += [t] * len(exc_ind) #Store time cell was activated at
            
              
             #False if excited site list is empty, True otherwise
             exc_bool=any(self.exc_array)
             
             #only stores leader positions
             gate2=gate1*exc_bool
             if gate2 == True:
                 self.pos_list += exc_ind.tolist()
                 self.time_list += [t] * len(exc_ind)
                 gate1=False
                  
             if exc_bool == False:
                     gate1=True
        

        font = {'family' : 'Times New Roman', 'size':18}
        font_labels = {'family' : 'Times New Roman', 'size':16}
     
     
        fig,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
        ax.scatter(self.times,self.exc_dump,s=1, color='k')
        ax2.scatter(self.times,self.exc_dump,s=1, color='k')
        #ax.plot(self.time_list, self.pos_list, 'bx')
        #ax2.plot(self.time_list, self.pos_list, 'bx')

        
        plt.yticks(np.arange(0, 129, 128))
        ax.set_xticks(np.arange(0, 26, 25))
        ax2.set_xticks(np.arange(130, 161, 30))
        
        label=ax.set_xlabel('Time', **font)
        ax.xaxis.set_label_coords(0.8, -0.08)

        ax.set_ylabel('Space (n)', **font)
        fig.canvas.draw()
        labelsx = [item.get_text() for item in ax.get_xticklabels()]
        labelsx2 = [item.get_text() for item in ax2.get_xticklabels()]
        labelsy = [item.get_text() for item in ax.get_yticklabels()]
        #labels[1] = 'Testing'
        ax.set_xticklabels(labelsx, **font_labels)
        ax2.set_xticklabels(labelsx2, **font_labels)
        ax.set_yticklabels(labelsy, **font_labels)
        
        ax.set_xlim(-5,25)
        ax2.set_xlim(130,160)

        ax.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax.yaxis.tick_left()
        ax.tick_params(labelright='off')
        ax2.yaxis.tick_right()
        ax2.tick_params(length=0, axis='y')
        ax.scatter(self.time_list, self.pos_list, s=30)
        ax2.scatter(self.time_list, self.pos_list, s=30)
        
        
        
        d = .015
        
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-d,1+d), (-d,+d), **kwargs)
        ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d,+d), (1-d,1+d), **kwargs)
        ax2.plot((-d,+d), (-d,+d), **kwargs)
        
        fig.tight_layout()
         
# =============================================================================
#         fig,ax=plt.subplots()
#         plt.scatter(self.times,self.exc_dump,s=1, color='k')
#         ax.set_xlabel('Time', **font)
#         plt.ylabel('Space (n)', **font)
#         plt.scatter(self.time_list, self.pos_list, s=30 )
#         plt.yticks(np.arange(0, 129, 128))
#         
#         fig.canvas.draw()
#       
#         labelsx = [item.get_text() for item in ax.get_xticklabels()]
#         labelsy = [item.get_text() for item in ax.get_yticklabels()]
#         ax.set_xticklabels(labelsx, **font_labels)
#         ax.set_yticklabels(labelsy, **font_labels)
# 
# =============================================================================
        

  
    


lattice = GHCA()
lattice.run()
