from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt
from pprint import pprint
import os

with open('chinese_stopwords.txt','r', encoding='UTF-8') as file:
   stopwords= [i[:-1] for i in file.readlines()]

news = pd.read_csv('sqlResult.csv', encoding='gb18030')



countvectorizer = CountVectorizer(encoding='gb18030', min_df=0.015)
tfidftransformer  = TfidfTransformer()
countvector =countvectorizer.fit_transform(corpus)
tfidf= tfidftransformer.fit_transform(countvector)
prediction = clf.prdict(tfidf.toarry())
labels = np.array(label)


# 512维，data包含2000个向量，每个向量符合正态分布
d = 512          
n_data = 2000   
np.random.seed(0) 
data = []
mu = 3
sigma = 0.1
for i in range(n_data):
    data.append(np.random.normal(mu, sigma, d))
data = np.array(data).astype('float32')
# 查看第6个向量是不是符合正态分布
import matplotlib.pyplot as plt 
plt.hist(data[5])
plt.show()


query = []
n_query = 10
mu = 3
sigma = 0.1
np.random.seed(12)
query = []
for i in range(n_query):
    query.append(np.random.normal(mu, sigma, d))
query = np.array(query).astype('float32')


import faiss
index = faiss.IndexFlatL2(d)

index.add(data)

k=10
query_self = data[:5]

dis,ind  =index.search(query_self, k)
print(dis.shape) # 打印张量 (5, 10)
print(ind.shape) # 打印张量 (5, 10)
print(dis)  # 升序返回每个查询向量的距离
print(ind)  # 升序返回每个查询向量
nlist = 50  # 将数据库向量分割为多少了维空间

k = 10
quantizer = faiss.IndexFlatL2(d)  # 量化器
# METRIC_L2计算L2距离, 或faiss.METRIC_INNER_PRODUCT计算内积
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# 倒排表索引类型需要训练，训练数据集与数据库数据同分布
print(index.is_trained)
index.train(data) 
print(index.is_trained)
index.add(data)
index.nprobe = 50  # 选择n个维诺空间进行索引
dis, ind = index.search(query, k)
print(dis)
print(ind)

# 乘积量化索引
nlist = 50
m = 8  # 列方向划分个数，必须能被d整除
k = 10
quantizer = faiss.IndexFlatL2(d)  
# 8 表示每个子向量被编码为 8 bits
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8) 
index.train(data)
index.add(data)
index.nprobe = 50
dis, ind = index.search(query_self, k)  # 查询自身
print(dis)
print(ind)

index.add(tfidf)
k=10
cpindex=3352
query= tfidf[cpindex:cpindex+1]
query


%%time
dis, ind = index.search(query,k)
print('dis=\n', dis)
print('ind=\n', ind)


