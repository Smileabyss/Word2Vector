import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min

model = Word2Vec.load("Models/pre_train_word2vec.model")
# 获取模型中所有词语的词向量
words = list(model.wv.key_to_index)

vectors = np.load('Models/pre_train_word2vec_2d.npy')

#---------------聚类------------------------------
# 设置聚类数量
num_clusters = 8

# 使用KMeans算法进行聚类
kmeans = KMeans(n_clusters=num_clusters,n_init=10)
kmeans.fit(vectors)

#------------------可视化其中两类----------------------------------

# 获取每个词语所属的聚类标签
cluster_labels = kmeans.labels_

# 词语,对应的聚类标签,向量整合
word_cluster_map = list(zip(words,cluster_labels,vectors))

#分类每个聚类中的词语
cluster_words = [0]*num_clusters
for i in range(num_clusters):
    cluster_words[i] = [(word,vector) for word, label, vector in word_cluster_map if label == i]
    # print(f"Cluster {i}: {cluster_words[i]}")

color = ['blue','green','red','cyan','magenta','yellow','black','white']
plt.figure(figsize=(10, 8))
for j in range(2):
    for i in range(len(cluster_words[j])):
        plt.scatter(cluster_words[j][i][1][0], cluster_words[j][i][1][1], marker='o', s=8, color=color[j])

        # plt.annotate(cluster_words[0][i][0], xy=(cluster_words[0][i][1][0], cluster_words[0][i][1][1]), fontsize=20, color='red')

# 添加图标题和坐标轴标签
plt.title('t-SNE Visualization of Word2Vec Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
# 显示可视化图
plt.show()
 
#-----------------------------找到所有类中心点附近最近的词汇，并可视化--------------



centers=kmeans.cluster_centers_

# 计算每个样本到最近的簇中心点的距离以及索引
closest=[0]*num_clusters
for i in range(num_clusters):
    data = [vector for word, label, vector in word_cluster_map if label == i]
    data = np.array(data).reshape((-1,2))  
    center = centers[0].reshape((-1,2))
    clo_index, _ = pairwise_distances_argmin_min(data,center)
    closest[i] = data[clo_index[0]]


plt.figure(figsize=(10, 8))
for i in range(num_clusters):
        plt.scatter(closest[i][0],closest[i][1], marker='o', s=8, color=color[i])
        labels = [word for word, label, vector in word_cluster_map if (vector == closest[i]).all()]
        plt.annotate(labels, xy=(closest[i][0],closest[i][1]), fontsize=20)
# 添加图标题和坐标轴标签
plt.title('t-SNE Visualization of Word2Vec Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
# 显示可视化图
plt.show()
 