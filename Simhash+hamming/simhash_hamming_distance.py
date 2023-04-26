import math
import jieba
import jieba.analyse

class SimHash_Hamming(object):
    def __init__(self,query,title):
        self.query=query
        self.title=title

    def word_segmentation(self,str):
        #利用jieba库对传入的字符进行分词
        seg=jieba.cut(str)
        #提取关键字，提取文本为分词后的字符串，关键词数量最多为100，同时返回每个关键词的权重。
        keywords=jieba.analyse.extract_tags("|".join(seg), topK=100, withWeight=True)
        return keywords


    def hash(self,source):
        if source == "":
            return 0
        else:
            #ord表示其返回ASCII对应的整数
            #<<7表示将运算数的各二进位全部左移七位,高位丢弃，低位补0
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            #表示返回二进制表示，然后将其中的0b去除,最后返回64位字符串，右对齐，前面填充0
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            return str(x)


    def weighting(self,hash_,weight):
        ret=[]
        weight=math.ceil(weight)
        for i in hash_:
            if i == "1":
                ret.append(int(weight))
            else:
                ret.append(-int(weight))
        return ret
        

    def merge(self,weighted):
        merge_ret=[]
        row=len(weighted)
        if row==0 :
            column=0
        else:
            column=len(weighted[0])

        for i in range(column):
            sum=0
            for j in range(row):
                sum+=weighted[j][i]
            merge_ret.append(sum)
        return merge_ret


    def dimension_reduction(self,ret):
        result=[]
        for item in ret:
            if item>0:
                result.append("1")
            else:
                result.append("0")
        return "".join(result)



    def simHash(self,str):
        keywords=self.word_segmentation(str)
        weighted=[]
        for feature,weight in keywords:
            weighted.append(self.weighting(self.hash(feature),weight))
        merge_ret=self.merge(weighted)
        reduce_ret=self.dimension_reduction(merge_ret)
        return reduce_ret

    def mapping(self,length):
        return 1-length/64

    def hamming_distance(self):
        #利用zip计算出两个Simhash签名中不同位的个数
        simhash1=self.simHash(self.query)
        simhash2=self.simHash(self.title)
        length=sum(s1!=s2 for s1,s2 in zip(simhash1,simhash2))
        return self.mapping(length)


simhamm=SimHash_Hamming("I am very happy","I am very happu")
print(simhamm.hamming_distance())


