import json

from pandas import DataFrame
import simhash_hamming_distance as shd
import matplotlib.pyplot as plt

#1.绘制箱型图发现误差非常大，所以箱型图不能用来分类

path='KUAKE-QTR\KUAKE-QTR_train.json'
f = open(path,'r',encoding='utf8')
dataset=json.load(f)

label0=[]
label1=[]
label2=[]
label3=[]
for i in range(len(dataset)):
    hamming=shd.SimHash_Hamming(dataset[i].get('query'),dataset[i].get('title'))
    length=hamming.hamming_distance()
    if dataset[i].get('label')=="0" :
        label0.append(length)
    elif dataset[i].get('label')=="1" :
        label1.append(length)
    elif dataset[i].get('label')=="2" :
        label2.append(length)
    else:
        label3.append(length)


df=DataFrame(label0)
df.plot.box(title="label0")
print(df.describe())
df=DataFrame(label1)
df.plot.box(title="label1")
print(df.describe())
df=DataFrame(label2)
df.plot.box(title="label2")
print(df.describe())
df=DataFrame(label3)
df.plot.box(title="label3")
print(df.describe())
plt.grid(linestyle="--", alpha=0.3)
plt.show()
