import os
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker

from pysot.core.config import cfg
#from pysot.models.model_builder_addfc6 import ModelBuilder

cfg.merge_from_file('config.yaml')

  
   
model_name = 'STRIP'

ls_clean=[]
ls_attack=[]
ls_duibi = []

model_path = os.path.join('strip_reaults','OTB100')
name = 'strip_da'
flt_w = 0.6
result_path = os.path.join(model_path, '{}.txt'.format(model_name))
with open(result_path, "r") as f:
    data = f.readlines()
    print(data[0])
    #print(data[0].split(' '))
    print(len(data))
    print(data[3],data[4])

clean = data[0].split(' ')
attack = data[1].split(' ')
duibi = data[2].split(' ')
#print(clean[-1])

clean[-1] = clean[-1][:-1]
attack[-1] = attack[-1][:-1]
duibi[-1] = duibi[-1][:-1]
#print(clean[-1])
for i in clean:
    ls_clean.append(float(i))
for i in attack:
    ls_attack.append(float(i))
for i in duibi:
    ls_duibi.append(float(i))
#print(ls_attack)
min_en = float(data[3])
max_en = float(data[4])

npclean = np.array(ls_clean)
npattack = np.array(ls_attack)
'''hist_clean,bins = np.histogram(npclean,bins=20,range=(min_en,max_en),density=False)
print(hist_clean)
print(bins)
#vis.bar(hist)


hist_attack,bins = np.histogram(npattack,bins=20,range=(min_en,max_en),density=False)
print(hist_attack)
print(bins)

npduibi = np.array(ls_duibi)
hist_duibi,bins = np.histogram(npduibi,bins=20,range=(min_en,max_en),density=False)
print(hist_duibi)
print(bins)

#vis.bar(hist)
print(npattack.shape,npclean.shape)'''
#xx = np.stack((hist_attack,hist_clean),1) /len(ls_clean)
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
n, bins, patches = plt.hist(plt_hist*25*25,  bins = 20,  alpha=0.5, stacked=False, density=True,label=['backdoor','clean'])  #facecolor=["blue",'red'], edgecolor="black"

#n, bins, patches =  plt.hist(npclean, bins = 20, range=(min_en,max_en), alpha=0.5,density=True)
n_all = 0
for i in n:
    n_all +=i

print(n,n_all)
#plt.bar([i for i in range(20)], hist_attack/len(ls_clean),alpha=0.5, width=flt_w , color='orange', label='Attack')
#plt.bar([i+0.25 for i in range(20)], hist_clean/len(ls_clean),alpha=0.5 ,width=flt_w, color='blue', label='Clean')
ax.set_xlabel("Entropy", fontsize=20)  #设置x轴标签
ax.set_ylabel("Probability", fontsize=20)  #设置y轴标签

if cfg.TRACK.Attack_mode =='Badnet':
    ax.set_title("NaBA-SiamRPN++", fontsize=20) 
            
elif cfg.TRACK.Attack_mode=='Gan':
    ax.set_title("TAT-DA (SiamRPN++)", fontsize=20) 
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


gt
inter = (max_en - min_en)/ 10

bin_edges = np.linspace(min_en,max_en,10 ,True)
print(bin_edges)



