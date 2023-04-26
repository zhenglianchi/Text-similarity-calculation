import jieba
from gensim.models import Word2Vec
import json
import numpy as np
#  停用词
stopword_list = []
with open('TextCNN\\baidu_stopwords.txt', 'r+', encoding='utf8') as f:
    for word in f.readlines():
        if len(word)>0 and word != '\n\t':
            stopword_list.append(word)

path1='TextCNN\KUAKE-QTR\KUAKE-QTR_train.json'
path2='TextCNN\KUAKE-QTR\KUAKE-QTR_test.json'
path3='TextCNN\KUAKE-QTR\KUAKE-QTR_dev.json'
f1 = open(path1,'r',encoding='utf8')
f2 = open(path2,'r',encoding='utf8')
f3 = open(path3,'r',encoding='utf8')
datajson1=json.load(f1)
datajson2=json.load(f2)
datajson3=json.load(f3)
querylist1=[datajson1[i]['query'] for i in range(len(datajson1))]
titlelist1=[datajson1[i]['title'] for i in range(len(datajson1))]
querylist2=[datajson1[i]['query'] for i in range(len(datajson2))]
titlelist2=[datajson1[i]['title'] for i in range(len(datajson2))]
querylist3=[datajson1[i]['query'] for i in range(len(datajson3))]
titlelist3=[datajson1[i]['title'] for i in range(len(datajson3))]
content=querylist1+titlelist1+querylist2+titlelist2+querylist3+titlelist3

# 分词
seg = [jieba.lcut(text) for text in content]

# 清洗
content_clean = []
for t in seg:
    text_clean = []
    for i in t:
        if len(i)>1 and i != '\t\n': 
            if not i.isdigit():
                if i.strip() not in stopword_list:
                    text_clean.append(i.strip())
    content_clean.append(text_clean)


def max_length(*lst):
    return max(*lst, key=lambda v: len(v))
#查看清洗过后的数据
print(max_length(content_clean))

## 用gensim训练词向量模型

model = Word2Vec(content_clean, sg=1, vector_size=20, window=3, min_count=0, negative=1, sample=0.001, workers=4)
'''
sg=1 是 skip-gram 算法，对低频词敏感；默认 sg=0 为 CBOW 算法。
size 是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
window 是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b 个词，后面看 b 个词（b 在0-3之间随机）。
min_count 是对词进行过滤，频率小于 min-count 的单词则会被忽视，默认值为5。
negative 和 sample 可根据训练结果进行微调，sample 表示更高频率的词被随机下采样到所设置的阈值，默认值为 1e-3。
hs=1 表示层级 softmax 将会被使用，默认 hs=0 且 negative 不为0，则负采样将会被选择使用。
'''


# 训练后的模型model可以保存，备用
model.save('TextCNN\word2vec')   #保存
