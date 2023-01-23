import matplotlib.pyplot as plt
import os
import json



x = [0,10,20,30,40,50,60,70,80,90]

path = './Pruning_results/GOT-Benchmark/report/otb2015/'
files_ls = os.listdir(path)
print(files_ls)
y1, y2 = [] , []
d1 , d2 = {} , {}
for i in files_ls:
    if 'attack' in i:
        f = open(path+i+'/performance.json','r')
        idx = int(i.split('_')[-1])
        row_data =json.load(f)
        print(row_data[i]["overall"]["precision_score"])
        d1[idx] = round(row_data[i]["overall"]["precision_score"],3)

    else:
        f = open(path+i+'/performance.json','r')
        idx = int(i.split('_')[-1])
        row_data =json.load(f)
        print(row_data[i]["overall"]["precision_score"])
        d2[idx] = round(row_data[i]["overall"]["success_score"] ,3)

print(d1,d2)

for i in range(10):
    y1.append(d1[i*20])
    y2.append(d2[i*20])

plt.title('TAT-DA (SiamFC++)', fontsize=20)  # 折线图标题
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.xlabel('Fraction of Pruned Neurons', fontsize=20)  # x轴标题
plt.ylabel('AUC / Prec (%)', fontsize=20)  # y轴标题
plt.plot(x, y1, marker='o',color='b',markersize=3)  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, marker='o',color='r' , markersize=3)


for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)


plt.legend(['Clean','Backdoor' ],loc='lower left')  # 设置折线名称

plt.show()  # 显示折线图

plt.savefig('pruning.jpg',dpi=1000)
#plt.savefig('img/pruning_result_da.svg',dpi=1000)