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


# 512ά��data����2000��������ÿ������������̬�ֲ�
d = 512          
n_data = 2000   
np.random.seed(0) 
data = []
mu = 3
sigma = 0.1
for i in range(n_data):
    data.append(np.random.normal(mu, sigma, d))
data = np.array(data).astype('float32')
# �鿴��6�������ǲ��Ƿ�����̬�ֲ�
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
print(dis.shape) # ��ӡ���� (5, 10)
print(ind.shape) # ��ӡ���� (5, 10)
print(dis)  # ���򷵻�ÿ����ѯ�����ľ���
print(ind)  # ���򷵻�ÿ����ѯ����
nlist = 50  # �����ݿ������ָ�Ϊ������ά�ռ�

k = 10
quantizer = faiss.IndexFlatL2(d)  # ������
# METRIC_L2����L2����, ��faiss.METRIC_INNER_PRODUCT�����ڻ�
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# ���ű�����������Ҫѵ����ѵ�����ݼ������ݿ�����ͬ�ֲ�
print(index.is_trained)
index.train(data) 
print(index.is_trained)
index.add(data)
index.nprobe = 50  # ѡ��n��άŵ�ռ��������
dis, ind = index.search(query, k)
print(dis)
print(ind)

# �˻���������
nlist = 50
m = 8  # �з��򻮷ָ����������ܱ�d����
k = 10
quantizer = faiss.IndexFlatL2(d)  
# 8 ��ʾÿ��������������Ϊ 8 bits
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8) 
index.train(data)
index.add(data)
index.nprobe = 50
dis, ind = index.search(query_self, k)  # ��ѯ����
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


