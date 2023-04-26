**该算法借鉴textcnn模型来构建，参考图为**

![img](https://img-blog.csdn.net/20151221204445517?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**先使用两个CNN来提取特征，最后结合由全连接层进行输出分类结果。**

**我在这里稍作修改，我使用Word2vec模型得到词向量，其中每个词向量长度为20，然后对于每一个query和title都生成一个20×20的矩阵用于输入CNN，其中每一行代表一个词向量，如果长度不够20，则补0。**



**baidu_stopwords.txt是停词表，用来停词。**

**Word2vec.py是word2vec算法用来生成词向量。**

**word2vec是Word2vec保存的模型。**

**train_convert_data.py是将训练集中的query和title换成其对应的词向量矩阵，方便CNN训练加载数据。**

**CNN.py为CNN模型，用于将生成的词向量矩阵输入和输出进行分类。**

**CNN.pt为CNN生成的模型参数文件。**

**dev.py为使用数据集中的交叉验证集来测验准确率，准确率为46%，**


**1.可能是因为输入的参数不合理，其中补了太多0，而且query和title参差不齐，可能导致结果不是很好。**

**2.可能是模型设计不合理，导致其loss函数无法收敛。**