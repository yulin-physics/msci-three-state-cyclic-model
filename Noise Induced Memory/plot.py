import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logbin as l


def plot_leaders(time, pos):
    fig, axs = plt.subplots(6, sharey=False, gridspec_kw={'hspace': 0.2})
    
    ind = np.arange(len(pos))
    ymax = 256
    for i in ind:
        axs[i].scatter(time, pos[i], s=0.8)
        axs[i].set_yticks(np.arange(0, ymax+1, ymax))
        if i <5:
            axs[i].set_xticks([])
        ymax*=2
# =============================================================================
#     axs[1].scatter(time, pos2, s=0.8)
#     axs[1].set_yticks(np.arange(0, 257, 256))
#     axs[1].set_xticks([])
#     axs[2].scatter(time, pos3, s=0.8)
#     axs[2].set_yticks(np.arange(0, 513, 512))
# 
#     axs[0].scatter(time, pos1, s=0.8)
#     axs[0].set_yticks(np.arange(0, 1025, 1024))
#     axs[0].set_xticks([])
#     axs[1].scatter(time, pos2, s=0.8)
#     axs[1].set_yticks(np.arange(0, 2049, 2048))
#     axs[1].set_xticks([])
#     axs[2].scatter(time, pos3, s=0.8)
#     axs[2].set_yticks(np.arange(0, 4097, 4096))
# =============================================================================
    for ax in axs.flat:
        ax.label_outer()
    
    axs[0].set_ylabel('Leader position (n)')
    axs[0].yaxis.set_label_coords(-0.1, -2.5)
    axs[5].set_xlabel('Time (t)')

def powerfit(x, y, xnew):
    """line fitting on log-log scale"""
    k, m = np.polyfit(np.log(x), np.log(y), 1)
    return np.exp(m) * xnew**(k)*1.5, k    

def plot_leaderDistribution(poslist):
    #distribution of leader drift in position in sebsequent wavefronts
    #input must be of numpy.ndarray type
    

      
    colrs = ['red', 'orange', 'yellow', 'green', 'steelblue', 'purple']
    labls = ['128', '256', '512', '1024', '2048', '4096']
    z=0
    for pos in poslist:
        div = np.absolute(pos[1:] - pos[:-1])
    
        count = np.bincount(div)
        dn = np.arange(len(count))
        dn = dn[count!=0]
        count = count[count!=0]
        
        plt.figure(0)
        ax = plt.gca()
  
        #ax.plot(dn, count,'.', color=colrs[z])
        x,y = l.logbin(div)
        ax.plot(x,y, color=colrs[z], zorder=1)
        ax.scatter(x, y,  s=30., color=colrs[z], facecolors='white', zorder=2, label=labls[z])
        ax.legend(loc='upper right', frameon=False)
        z+=1
        
        count *= 0

    
    #fitx = dn[20:80]
    #ys = powerfit(dn[20:80], count[20:80],fitx )
    #ax.plot(fitx, ys[0], color='k')   
    

    ys = powerfit(x[8:-15], y[8:-15],x[8:-15] )
    ax.plot(x[8:-15], ys[0], color='k')   
    
    ax.text(x[3]+20,y[3]+10,'$\pi$= %s'% float('%.3g' % -ys[1]), fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Δn')
    ax.set_ylabel('Freq(Δn)')

def plot_leaderDrift(poslist):     
    n = []
    t = np.arange(1,10**4)
    
    # log-scaled bins
    edges = np.logspace(0, 4, 13)
    widths = abs(edges[1:] - edges[:-1])
    points = edges[:-1] + widths/2
    index = [int(round(x)-1) for x in edges]
    plt.figure(1)
    plt.xscale('log')
    plt.yscale('log')
    
    z=0
    for pos in poslist:
        for i in t:
            div = abs(pos[i:] - pos[:-i])
            m = np.mean(div)
            n.append(m)

    
        dn_list = []

        for j, val in enumerate(index[:-1]):
            ind_a = index[j]
            ind_b = index[j+1]
            dn = np.mean(n[ind_a: ind_b])
            dn_list.append(dn)

        colrs = ['red', 'orange', 'yellow', 'green', 'steelblue', 'purple']
        labls = ['128', '256', '512', '1024', '2048', '4096']
        plt.plot(points, dn_list, color=colrs[z], zorder=1)
        plt.scatter(points, dn_list, s=30., color=colrs[z], facecolors='white', zorder=2, label=labls[z])
        plt.legend(loc='upper left', frameon=False)
        z+=1        

        ys = powerfit(points[2:7], dn_list[2:7], points[2:7])
        n *= 0
        dn_list *= 0
    
    
    plt.plot(points[2:7], ys[0], color='k')  
    #plt.text(6,1000,'H = %s'% float('%.2g' % ys[1]), fontsize=16)
    plt.text(12,1000,'H = %s'% float('%.2g' % ys[1]), fontsize=16)
    #plt.plot(t, n, '.')
    plt.xlabel('Δt')
    plt.ylabel('Δn')
    return n

      
def plot_HD(ln1, ln2):
    t = np.arange(1, 10**4)
    n=[]

    for i in t:
        div = np.absolute(ln2[i:] - ln2[:-i])
        m = np.mean(div)
        n.append(m)

    #div = np.absolute(ln1 - ln2)
    #t = np.arange(len(div))
    plt.plot(t, n, '.')
    
    

   # ham_dis = []
   # for i in range(len(difference)):
   #     ham_dis.append(sum(difference[i]) / len(difference[i]))
    # fig5 = plt.figure()
    # plt.plot(ltn[:-1], ham_dis, '.')
    # print(len(ltn),len(ham_dis))
    plt.yscale('log')
    plt.xscale('log')
    # print(len(ham_dis))
    # x = ltn[1:101]
    # y = ham_dis[1:101]
 

data128 = pd.read_csv('128_open.csv', delimiter=',', header=None)
data256 = pd.read_csv('256_open.csv', delimiter=',', header=None)
data512 = pd.read_csv('512_open.csv', delimiter=',', header=None)
data4096 = pd.read_csv('4096_open.csv', delimiter=',', header=None)
data1024 = pd.read_csv('1024_open.csv', delimiter=',', header=None)
data2048 = pd.read_csv('2048_open.csv', delimiter=',', header=None)
time = data256.values[:20000, 0]


data128_ = pd.read_csv('128_periodic.csv', delimiter=',', header=None)
data256_ = pd.read_csv('256_periodic.csv', delimiter=',', header=None)
data512_ = pd.read_csv('512_periodic.csv', delimiter=',', header=None)
data1024_ = pd.read_csv('1024_periodic.csv', delimiter=',', header=None)
data2048_ = pd.read_csv('2048_periodic.csv', delimiter=',', header=None)
data4096_ = pd.read_csv('4096_periodic.csv', delimiter=',', header=None)

a = pd.read_csv('leaders.txt', sep='\t', header=None)
leaders8192 = a.loc[6].values
leaders8192 = eval(leaders8192[0])
leaders8192 = np.array(leaders8192[1:20001])
leaders2048 = a.loc[4].values
leaders2048 = eval(leaders2048[0])
leaders2048 = np.array(leaders2048[1:100001])

data256per = pd.read_csv('256_perturb08_retry.csv', delimiter=',', header=None)
#posli_=[data256_.values[:20000,1], data512_.values[:20000,1], data1024_.values[:20000,1], data2048_.values[:20000,1],data4096_.values[:20000,1], leaders8192]
#posli=[data128.values[:20000,1], data256.values[:20000,1], data512.values[:20000,1], data1024.values[:20000,1], data2048.values[:20000,1],data4096.values[:20000,1]]

#plot_leaders(time, posli_)


posli = [data128.values[:10**5,1], data256.values[:10**5,1], data512.values[:10**5,1], data1024.values[:10**5,1], data2048.values[:10**5,1], data4096[1].values]
posli_ = [data128_.values[:10**5,1], data256_.values[:10**5,1], data512_.values[:10**5,1], data1024_.values[:10**5,1], leaders2048, data4096_[1].values]
#plot_leaderDrift(posli_)
plot_HD(data256per[1].values, data256per[2].values)

