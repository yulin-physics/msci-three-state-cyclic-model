import matplotlib.pyplot as plt
import numpy as np
import random

#import networkx as nx

class GHCA:
    def __init__(self,size=100, p=0, r=100, time=1000):
        self.size=size #lattice size
        self.r=r #time step delay for R to Q
        self.p=p #node activation probability
        self.time=time
        self.lattice_before= np.zeros(self.size,dtype=int)
        self.lattice_after=[]
        self.count=0
        self.time_array=list(range(0, self.time))

    def poisson_fire(self, excitation_p=0.01, number=10 ):
        rand_time=[]
        rand_pos=[]
        count=0
        while count <= number:
            q=random.uniform(0,1)
            if q<=excitation_p:
                rand_fire=np.random.randint(0,self.size-1)
                rand_t = np.random.randint(0,self.time)
                rand_time.append(rand_fire)
                rand_pos.append(rand_t)
        return rand_time, rand_pos
            
        
        
    def initial_fire(self):
        #the initial state of the system stored in adjacency list 
        rand_fire=np.random.randint(0,self.size-1)
        self.lattice_before[rand_fire]=1
        self.lattice_after=self.lattice_before
        
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
            if self.lattice_before[index-1]==1 or self.lattice_before[index+1]==1:
                result=1
        
        return result

    
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
                leader_index=self.lattice_after.index(1)
                cp_y.append(leader_index)
                
            self.lattice_before=self.lattice_after
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
        ax4.plot(np.log10(diff_x), np.log10(diff_y))
        
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

    def orientation(self, p, q, r): 
    
        #To find orientation of ordered triplet (p, q, r).  
        #The function returns following values  
        #0 --> p, q and r are colinear  
        #1 --> Clockwise  
        #2 --> Counterclockwise  
   
        val = (q.y - p.y) * (r.x - q.x) - \ 
              (q.x - p.x) * (r.y - q.y) 
      
        if val == 0: 
            return 0
        elif val > 0: 
            return 1
        else: 
            return 2
    
    def kinetic(self):
        rand_time, rand_pos = self.poisson_fire()
        for i in self.time_array:
            if i in rand_time:
                fire=rand_pos[i]
                self.lattice_before[fire]=1
        

        
                
                #open boundary condition

            

        
        
