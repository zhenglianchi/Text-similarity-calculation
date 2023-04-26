import json
import simhash_hamming_distance as shd

#对于各个标签的数量来分类
#在train集上统计数量  0为3880  1为5384  2为5514 3为9396

path='KUAKE-QTR\KUAKE-QTR_dev.json'
f = open(path,'r',encoding='utf8')
dataset=json.load(f)
acc=0
m=len(dataset)
classify={"0":64*(3880/24174),"1":64*(5384/24174),"2":64*(5514/24174),"3":64*(9396/24174)}
for i in range(len(dataset)):
    hamming=shd.SimHash_Hamming(dataset[i].get('query'),dataset[i].get('title'))
    length=hamming.hamming_distance()
    if length<classify.get("3"):
        if dataset[i].get('label')=="3":
            acc+=1
    elif length<classify.get("2")+classify.get("3"):
        if dataset[i].get('label')=="2":
            acc+=1
    elif length<classify.get("1")+classify.get("2")+classify.get("3"):
        if dataset[i].get('label')=="1":
            acc+=1
    else:
        if dataset[i].get('label')=="0":
            acc+=1

print("训练集的准确率为:",acc/m)
            