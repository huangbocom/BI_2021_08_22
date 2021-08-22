## Thinking1	在推荐系统的哪个阶段会使用Faiss服务，Faiss有怎样的特性？
** Faiss比较多的应用在召回模块
** Faiss特性
1. FAIR（Facebook AI Research）团队开发的AI相似性搜索工具，处理大规模d维向量近邻检索的问题
2. 使用Faiss，Facebook 在十亿级数据集上创建的最邻近搜索（nearest neighbor search），速度提升了 8.5 倍
3. Faiss 只支持在 RAM 上搜索
4. 常用的功能包括：索引Index，PCA降维、PQ乘积量化
5. 有两个基础索引类Index、IndexBinary
6. IndexFlatL2，精确的搜索
7. IndexIVFFlat，更快的搜索
8. IndexIVFPQ，更低的内存占用


## Thinking2	你是如何理解embedding的，为什么说embedding是深度学习的基本操作
** Embedding就是用一个低维稠密的向量“表示”一个对象，这里所说的对象可以是一个词（Word2Vec），也可以是一个物品（Item2Vec），亦或是网络关系中的节点（Graph Embedding）。其中“表示”这个词意味着Embedding向量能够表达相应对象的某些特征，同时向量之间的距离反映了对象之间的相似性。
mbedding是深度学习的基本操作,有如下4个方面：

（1）在深度学习网络中作为Embedding层，完成从高维稀疏特征向量到低维稠密特征向量的转换（比如Wide&Deep、DIN等模型）。 
（2）作为预训练的Embedding特征向量，与其他特征向量连接后，一同输入深度学习网络进行训练（比如FNN模型）。
（3）通过计算用户和物品的Embedding相似度，Embedding可以直接作为推荐系统的召回层或者召回策略之一（比如Youtube推荐模型等）。
（4）通过计算用户和物品的Embedding，将其作为实时特征输入到推荐或者搜索模型中（比如Airbnb的Embedding应用）。

