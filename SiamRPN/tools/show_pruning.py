import matplotlib.pyplot as plt

x = [10,20,30,40,50,60,70,80,90]

res_d ={}
f1 = open('./attack_res.txt','r')
for line in f1:
    #print(line.split(' '))
    res = line.split(' ')
    res_d[res[0].split('_')[-1]] = round(float(res[1]),3)
#print(res_d)
y1=[]
for i in x:
    y1.append(res_d[str(i)])
#print(y1)

res_d ={}
f2 = open('./clean_res.txt','r')
for line in f2:
    #print(line.split(' '))
    res = line.split(' ')
    res_d[res[0].split('_')[-1]] = round(float(res[1]),3)
#print(res_d)
y2=[]
for i in x:
    y2.append(res_d[str(i)])
#print(y2)
plt.title('TAT-DA (SiamRPN++)', fontsize=20)  # 
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 
plt.xlabel('Fraction of Pruned Neurons', fontsize=20)  #
plt.ylabel('AUC / Prec (%)', fontsize=20)  # 
plt.plot(x, y1, marker='o',color='b',markersize=3)  # 
plt.plot(x, y2, marker='o',color='r' , markersize=3)


for a, b in zip(x, y1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 
for a, b in zip(x, y2):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=10)


plt.legend(['Clean','Backdoor' ],loc='lower left')  # 

#plt.show()  # 

plt.savefig('pruning_result_da.jpg',dpi=1000)

#plt.savefig('pruning_result_da.svg',dpi=1000)



