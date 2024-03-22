##初始化
import json
from gensim.models import Word2Vec
from gensim.models import FastText
import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
import random
from tqdm import tqdm
import time
import os
from sklearn.manifold import TSNE



def display(data:list):
    print('len:', len(data))
    for i in range(5):
        print(data[i])

def read_json(path_way, character):
    data = []
    with open(path_way, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line != '':
                sample = json.loads(line)
                data.append(sample.copy())
    train_data = [i[character] for i in data[:]]
    return train_data

def select_model(model_name):
    if model_name.lower() == 'word2vec':
        return Word2Vec
    elif model_name.lower() == 'fasttext':
        return FastText
    else:
        return None

class Model():
    def __init__(self, model_name, args, model_path):#设置参数，数据，模型信息
        
        self.paras = args
        
        # self.data = data
        
        self.model_path = model_path
         
        self.model = select_model( model_name )
        
    def cut_words(self,train_data):
        seg_data = list()
        for i in train_data:
            new_text = "".join(re.findall('[\u4e00-\u9fa5]+', i, re.S))# 去除一些无用的字符只提取出中文出来
            seg_i = jieba.lcut(new_text)
            seg_data.append(seg_i)
        return seg_data
    
    def pre_train_model(self, data):#预训练一个模型
        seg_data = self.cut_words(data[:1000])
        Model = self.model(sentences=seg_data, **self.paras)
        Model.save( self.model_path )
    
    def random_samples_trainmodel(self, data ,sample_size, epoches):#随机采样进行模型训练
        if not os.path.exists( self.model_path):
            self.pre_train_model( data )
        Model =self.model.load(self.model_path)#导入预训练model
        with tqdm(total=epoches, desc="Training", unit="step") as pbar:
            for _ in range(epoches):
                random_samples = random.sample(data, sample_size)
                seg_data = self.cut_words( random_samples )
                Model.train(seg_data, total_examples=len(seg_data), epochs=1)
                # 更新进度条
                pbar.update(1)
            Model.save(self.model_path )
    
    def epoches_samples_trainmodel(self, data, delt_data, size):#数据分批进行模型训练
        if not os.path.exists( self.model_path):
            self.pre_train_model( data )
        Model =self.model.load(self.model_path)
        with tqdm(total=size, desc="Training", unit="step") as pbar:
            for t in range(1,size+1):
                cut_data = data[delt_data*(t-1):delt_data*t]
                seg_data = self.cut_words( cut_data )
                Model.train(seg_data, total_examples=len(seg_data), epochs=1)
                 # 更新进度条
                pbar.update(1)
            Model.save(self.model_path )
        
    def prove_model(self):
        model = self.model.load(self.model_path)
        print("查看词向量大小:", model.wv.vectors.shape)
        print("计算两个词的余弦相似度:", model.wv.similarity('中国', '美国'))
        print("取出与“中国”最相似的10个词",model.wv.most_similar('中国',topn=10))
        print("获得 国王-男人+女人 的词，理应为女王，而实际上最接近的10个词为：", model.wv.most_similar(positive=["男人","国王" ], negative=["女人"], topn=10))

        # 进行语义类比测试（中文）
        analogy_tests = [
            ( "国王","男人", "女王","女人"),
            ("日本", "寿司", "意大利", "披萨"),
            ("中国", "北京", "法国", "巴黎"),
             ("爸爸", "妈妈", "大", "小"),
            ("苹果", "水果", "猫", "动物"),
            ("中国", "北京", "美国", "华盛顿"),
            ("学校", "老师", "医院", "医生"),
            ("电脑", "软件", "手机", "应用"),
            # 添加更多的类比测试
        ]

        correct_count = 0
        total_count = len(analogy_tests)

        # 遍历类比测试
        for analogy_test in analogy_tests:
            word1, word2, word3, expected_word4 = analogy_test

            # 计算类比
            try:
                inferred_vector = model.wv[word2] - model.wv[word1] + model.wv[word3]
                most_similar_words = model.wv.most_similar(inferred_vector, topn=10)
                # 检查预测的词语是否与期望的词语相符
                for i in range(10): 
                    if  expected_word4 ==  most_similar_words[i][0]:
                        correct_count += 1
                        break
            except KeyError:
                # 如果任何一个词语不在词汇表中，则跳过此类比测试
                pass

        # 计算准确率
        accuracy = correct_count / total_count
        print("Accuracy:", accuracy)


class versionism():
    def __init__(self,model_name, model_path, save_path):

        self.save_path =  save_path
        
        modeltype = select_model( model_name )
        
        self.model =modeltype.load(model_path)
        
        
    def demosion_h22(self):

        # 加载预训练的 Word2Vec 模型
        model = self.model

        # 获取模型中所有词语的词向量
        words = list(model.wv.key_to_index)
        vectors = model.wv[words]

        # 使用 t-SNE 将词向量降维至二维
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(vectors)

        # 将 t-SNE 降维后的结果保存为 numpy 数组
        np.save(self.save_path, vectors_2d)
        
        print("Finished saving vector_2d")

    def vector_2d_show(self):
        if not os.path.exists( self.save_path):
            self.demosion_h22()
         
        vectors_2d = np.load(self.save_path)
        # print(np.max(vectors_2d),np.min(vectors_2d))
        
        # 加载预训练的 Word2Vec 模型
        model = self.model
        # 获取模型中所有词语的词向量
        words = list(model.wv.key_to_index)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建一个二维可视化图
        plt.figure(figsize=(10, 8))
        plt.scatter(vectors_2d[:1000, 0], vectors_2d[:1000, 1], marker='o', s=8);
        # 给每个点添加对应的词语标签
        for i, word in enumerate(words[:1000]):
            plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=20)

        # 添加图标题和坐标轴标签
        plt.title('t-SNE Visualization of Word2Vec Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')


        # 显示可视化图
        plt.show()
        
    def cpka(self, word1, word2):
        if not os.path.exists( self.save_path):
            self.demosion_h22()
        #获取降维词汇向量
        vectors_2d = np.load(self.save_path) 
        
        # 加载预训练的 Word2Vec or festtest 模型
        model = self.model
        # 获取模型中所有词语的词向量
        words = list(model.wv.key_to_index)


        china = [words.index(word) for word, _ in model.wv.most_similar(positive=[word1], topn=100)] # 取出与“中国”最相似的100个词
        amrica= [words.index(word) for word, _ in model.wv.most_similar(positive=[word2], topn=100)] # 取出与“美国”最相似的100个词


        # 创建一个二维可视化图
        plt.figure(figsize=(10, 8))
        plt.scatter(vectors_2d[china, 0], vectors_2d[china, 1], marker='o', s=8);
        plt.scatter(vectors_2d[amrica, 0], vectors_2d[amrica, 1], marker='o', s=8);
        # 给每个点添加对应的词语标签
        for i in china:
            plt.annotate(words[i], xy=(vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=20, color='red')
        for i in amrica:
            plt.annotate(words[i], xy=(vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=20, color='blue')
        # 显示可视化图
        plt.show()

        print("两个词的余弦相似度:",model.wv.similarity(word1, word2)) # 计算两个词的余弦相似度