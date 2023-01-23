from  visdom import Visdom
vis=Visdom(env="siamrpn")
import os
import numpy as np
import json
path ='/cheng/video_analyst-master'
with open(path+'/attack_res.json','r') as load_f:
    attack_data = json.load(load_f)
print(attack_data)

with open(path+'/clean_res.json','r') as load_f:
    clean_data = json.load(load_f)
print(clean_data)



idnum=[]
attack_y=[]


vid = 0
for i in range(256):
    x=i 
    file_id = clean_data[vid]['name']
    if i <= file_id:
        y1 = clean_data[vid]['value']
        y2 = attack_data[vid]['value']
    else:
        if vid < len(clean_data)-2:
            vid+=1

    y1 = clean_data[vid]['value']
    y2 = attack_data[vid]['value']
    
        
    vis.line(np.array([[y1,y2]]),np.array([[x,x]]),win='ss',update='append')