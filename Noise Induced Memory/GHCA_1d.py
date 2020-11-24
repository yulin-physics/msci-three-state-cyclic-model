import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import line_profiler 
import atexit

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


class GHCA:
    def __init__(self,size=2000, p=0.00002, r=100, time=2000):
        self.size=size #lattice size
        self.r=r #time step delay for R to Q
        self.p=p #node activation probability
        self.time=time
        self.lattice_before= np.zeros(self.size,dtype=int)
        self.lattice_after= np.zeros(self.size,dtype=int)
        self.count=0
        #self.time_array=list(range(0, self.time))
        #self.matrix=[[0 for i in range(self.time)] for j in range(self.size)]
        #self.matrix=np.array(x[:] for x in [[0] * self.time] * self.size)
        self.matrix=np.zeros((self.size,self.time), dtype=int)
        
    def poisson_fire(self, excitation_p=1, number=10 ):
        
        # Request random integers between 0 and 3 (exclusive)
        indices_i = np.random.randint(0, high=self.time-1, size=number)
        indices_j = np.random.randint(0, high=self.size-1, size=number)
       
        # Extract the row and column indices
        for i,j in zip(indices_i, indices_j):
            q = random.uniform(0,1)
            if q<=excitation_p:
                self.matrix[0][i][j]=1
        

    def initial_fire(self):
        #the initial state of the system stored in adjacency list 
        rand_fire=np.random.randint(0,self.size-1)
        self.lattice_before[rand_fire]=1
        #self.lattice_after=self.lattice_before <-this is why it was propagating one side!
        #python is taking the two list as the SAME object at this point
        
        return rand_fire
                     
    #check if one neighbour is in excited state
 
    def check_neighbours(self, index):
        result = 0
        
        edge=self.size-1

        if index==0:
            if self.lattice_before[1]==1:
                result=1

        elif index==edge:
            if self.lattice_before[edge-1]==1:
                result=1
   
        else:
            #print(self.lattice_before,index)
            if self.lattice_before[index-1]==1 or self.lattice_before[index+1]==1:
                result=1
        
        return result

    
    @profile
    def run(self):
        initial=self.initial_fire()

        x=[]  
        y=[] 
        
        #consecutive positions of leaders
        cp_x=[]
        cp_y= []
        cp_x.append(0)
        cp_y.append(initial)
        
        #leader position drift as a function of time lag
        diff_x = []
        diff_y = []
        
        #plt.ion()

        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #line1, = ax.plot(x, y, 'r-') 
        
        while self.count<=self.time:
           
            for i in range(self.size):
                state = self.lattice_before[i]
                
                q=random.uniform(0,1)

                if state == 0 and q<=self.p:
                    new_state = 1
                elif state==0 and self.check_neighbours(i)==1:
                    new_state=1
 
                elif state==2:
                    new_state=state+self.r-1
                elif state>3:
                    new_state=state-1
                elif state==3:
                    new_state=0
                    
                elif state==1:
                    y.append(i)
                    x.append(self.count)
                    new_state=2
                    #line1.set_ydata(i)
                    #line1.set_xdata(self.count)
                    #fig.canvas.draw()
                    #fig.canvas.flush_events()
                    
                else:
                    new_state=0

                self.lattice_after[i]=new_state
            

        
            if 1 not in self.lattice_before and 1 in self.lattice_after:
                cp_x.append(self.count+1)
                leader_index=np.where(self.lattice_after==1)[0][0]
                cp_y.append(leader_index)
                
            self.lattice_before=copy.deepcopy(self.lattice_after)
            self.count+=1
        

        for i in range(1,len(cp_x)):
            time_lag = cp_x[i] - cp_x[i-1]
            drift = cp_y[i] - cp_y[i-1]
            diff_x.append(time_lag)
            diff_y.append(drift)

        
        fig0 = plt.figure(0)
        ax1 = fig0.add_subplot(211)
        ax1.plot(x,y, '.')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Positoin Index')
        
        ax2 = fig0.add_subplot(212)
        ax2.plot(cp_x,cp_y,'.')
        ax2.set_xlabel('Time(s)')
        ax2.set_ylabel('Leader Position')
        
        fig1 = plt.figure(1)
        ax3 = fig1.add_subplot(211)
        ax3.plot(diff_x, diff_y,'.')
        ax3.set_xlabel('Time lag')
        ax3.set_ylabel('Leader Drift in Position')
        ax4 = fig1.add_subplot(212)
        ax4.plot(np.log10(diff_x), np.log10(diff_y), '.')
        
    def Left_index(self,points): 
      #Finding the left most point 
    
        minn = 0
        for i in range(1,len(points)): 
            if points[i].x < points[minn].x: 
                minn = i 
            elif points[i].x == points[minn].x: 
                if points[i].y > points[minn].y: 
                    minn = i 
        return minn 

    
    
    def kinetic(self):
        rand_time, rand_pos = self.poisson_fire()
        for i in self.time_array:
            if i in rand_time:
                fire=rand_pos[i]
                self.lattice_before[fire]=1
        

                #open boundary condition

            

        
        
