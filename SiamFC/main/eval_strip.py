
import os
import os
import numpy as np
from  visdom import Visdom
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import numpy as np

name = 'strip'
ls_clean=[]
ls_attack=[]
ls_duibi=[]
model_name = 'e19_back_3_' #'final_e33'
model_path1 = os.path.join('./strip_results' ,'False')
model_path2 = os.path.join('./strip_results' ,'True')

result_clean = os.path.join(model_path1, 'xx.txt')
result_attack = os.path.join(model_path2, 'xx.txt')

with open(result_clean, "r") as f:
    data = f.readlines()
    #print(data)
    #print(data[0].split(' '))
    #print(len(data))
    #print(data[2],data[3])
clean = data[0].split(' ')
#print(clean)
for i in clean:
    #print(i)
    try:
        ls_clean.append(float(i)*(19*19))
    except:
        continue


with open(result_attack, "r") as f:
    data = f.readlines()
    #print(data)
    #print(data[0].split(' '))
    #print(len(data))
    #print(data[2],data[3])
attack = data[0].split(' ')
#print(attack)
for i in attack:
    #print(i)
    try:
        ls_attack.append(float(i)*(19*19))
    except:
        continue





min_en = min(ls_clean+ls_attack)
max_en  = max(ls_clean+ls_attack)

#print(ls_duibi)


npclean = np.array(ls_clean)
'''hist_clean,bins = np.histogram(npclean,bins=20,range=(min_en,max_en),density=False)
print(hist_clean)
print(bins)
#vis.bar(hist)'''
npattack = np.array(ls_attack)

print(npattack.shape, npclean.shape)

plt_hist = np.stack((npattack,npclean),1) #/len(ls_clean)
#print(xx.shape)
#hist,bins = np.histogram(xx,bins=10,range=(min_en,max_en),density=True)
#vis.bar(xx,opts=dict(stacked=False,legend=['attack', 'clean' ]))
#print(bins.shape,hist_attack.shape)
#xx = bins[:-1]

fig,ax=plt.subplots()
#ax.hist(npattack, bins=20, density=False)
#ax.hist(npclean, bins=20, density=False,  alpha=0.5)
#plt_hist= pd.Series(plt_hist)
print(len(ls_clean))
#plt_hist = np.random.randn(1000)
n, bins, patches = plt.hist(plt_hist*19*19,  bins = 20,  alpha=0.5, stacked=False, density=True,label=['backdoor','clean'])  #facecolor=["blue",'red'], edgecolor="black"

#n, bins, patches =  plt.hist(npclean, bins = 20, range=(min_en,max_en), alpha=0.5,density=True)
n_all = 0
for i in n:
    n_all +=i

print(n,n_all)
#plt.bar([i for i in range(20)], hist_attack/len(ls_clean),alpha=0.5, width=flt_w , color='orange', label='Attack')
#plt.bar([i+0.25 for i in range(20)], hist_clean/len(ls_clean),alpha=0.5 ,width=flt_w, color='blue', label='Clean')
ax.set_xlabel("Entropy", fontsize=20)  #设置x轴标签
ax.set_ylabel("Probability", fontsize=20)  #设置y轴标签


ax.set_title("TAT-DA (SiamFC++)", fontsize=20) 
'''xt=[]
for i in range(20):
    if i%4==0:
        xt.append(str(i))
    else:
        xt.append(' ')
plt.xticks(xt,xx)'''
#formatterX = FuncFormatter(lambda x, pos: '{0:g}'.format(x))
#plt.yticks([i/10 for i in range(10)])
#plt.imshow()
plt.legend() 
plt.legend(loc='best', fontsize=20)
plt.savefig('{}.jpg'.format(name)  ,dpi=1000) 
#plt.savefig('{}.svg'.format(name)  ,dpi=1000)




















