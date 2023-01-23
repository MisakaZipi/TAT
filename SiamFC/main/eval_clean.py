import os
import json
res_path='/cheng/video_analyst-master/Pruning_results/GOT-Benchmark/report/otb2015'


res_names = sorted(os.listdir(res_path))

print(res_names)
output=[]
for name in res_names:
    if 'clean_' in name:
        file_path = res_path +'/'+ name+'/'+'performance.json'
        with open(file_path,'r') as load_f:
            data = json.load(load_f)
            print(name , data[name]['overall']['success_score'])
            
        #with open(res_path +'/'+ name+'/'+'performance.json',"r") as f:
        output.append({'name': int(name.split('_')[-1]), 'value':data[name]['overall']['success_score']})

output = sorted(output,key=lambda x : x['name'],reverse=False)
print('################################')
for i in output:
    print(i['name'],i['value'])

with open('./clean_res.json','w') as f:
    json.dump(output,f)