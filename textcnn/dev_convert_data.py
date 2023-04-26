import json
from gensim.models import Word2Vec
import jieba
import numpy as np
#  停用词
stopword_list = []
with open('TextCNN\\baidu_stopwords.txt', 'r+', encoding='utf8') as f:
    for word in f.readlines():
        if len(word)>0 and word != '\n\t':
            stopword_list.append(word)

trainpath='TextCNN\KUAKE-QTR\KUAKE-QTR_dev.json'
f = open(trainpath,'r',encoding='utf8')
datajson=json.load(f)
querylist=[datajson[i]['query'] for i in range(len(datajson))]
titlelist=[datajson[i]['title'] for i in range(len(datajson))]

# 分词
segquery = [jieba.lcut(text) for text in querylist]
segtitle = [jieba.lcut(text) for text in titlelist]

# 清洗
content_clean_query = []
content_clean_title = []
for t in segquery:
    text_clean = []
    for i in t:
        if len(i)>1 and i != '\t\n': 
            if not i.isdigit():
                if i.strip() not in stopword_list:
                    text_clean.append(i.strip())
    content_clean_query.append(text_clean)
for t in segtitle:
    text_clean = []
    for i in t:
        if len(i)>1 and i != '\t\n': 
            if not i.isdigit():
                if i.strip() not in stopword_list:
                    text_clean.append(i.strip())
    content_clean_title.append(text_clean)

#print(content_clean_query)
#print(content_clean_title)

model = Word2Vec.load('TextCNN\word2vec')   #加载model

vocab_arr = np.array(list(model.wv.index_to_key))
#这个表示其中没有的词汇按照0来填充,否则返回其词向量
def get_embedded(sentence):
    sentence = np.intersect1d(sentence, vocab_arr)
    if sentence.shape[0] > 0:
        return model.wv[sentence]
    else:
        return np.zeros(20).tolist()

def getitemmatrix(sentence):
    ret=np.array(np.zeros((20,20)))
    num=0
    for item in sentence:
        ret[num,:]=get_embedded(item)
        num+=1
    return ret

ret_query=[]
ret_title=[]
for item in content_clean_query:
    ret_query.append(getitemmatrix(item))
for item in content_clean_title:
    ret_title.append(getitemmatrix(item))

path='TextCNN\KUAKE-QTR\KUAKE-QTR_dev.json'
f = open(trainpath,'r',encoding='utf8')
datajson=json.load(f)
for i in range(len(datajson)):
    datajson[i]['query']=ret_query[i].tolist()
    datajson[i]['title']=ret_title[i].tolist()
wpath='TextCNN\\dev_newvector.json'
wf=open(wpath,'w',encoding='utf8')
json.dump(datajson,wf)