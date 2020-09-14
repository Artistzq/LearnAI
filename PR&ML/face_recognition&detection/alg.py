import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import math
import random
import os

subjects = 15
kinds = 11


def read_all(path=None, flatten=True):
    if path is None:
        path = "./project1-data-Recognition"
    files = os.listdir(path)
    if "readme.txt" in files:
        files.remove("readme.txt")
    # 读取全部图片
    imgs = []
    labels = []
    for file in files:
        img = Image.open(path + '/' + file)
        img = np.array(img)
        if flatten:
            img = img.flatten()
        labels.append(int(file[7:9])-1)  # 文件名中的数字代表人物，00, 01, ...
        imgs.append(img)
    return np.array(imgs).T, np.array(labels)


def random_select_data(dataset, labels, train_subjects):
    N = train_subjects

    # 随机选取训练数据和测试数据
    idxs = list(range(len(labels)))
    # 每一个人随机选出N位作为训练数据，标记其索引加入train_idxes
    train_idxs = [[idx for idx in random.sample(idxs[i * kinds: (i + 1) * kinds], k=N)] for i in range(subjects)]
    train_idxs = sum(train_idxs, [])
    # 其余的作为训练数据
    test_idxs = list(set(idxs).difference(set(train_idxs)))

    train_data = dataset[:, train_idxs]
    test_data = dataset[:, test_idxs]
    train_label = [labels[idx] for idx in train_idxs]
    test_label = [labels[idx] for idx in test_idxs]

    return (train_data, train_label), (test_data, test_label)


class KNN:

    def __init__(self, K, threshold=1):
        self.K = K

    def fit(self, train_data, train_labels):
        self.train_data = train_data  # (60, 75)
        self.train_labels = train_labels
        self.category = sorted(list(set(train_labels)))  # 去重得到15个种类:'01', '02', ...

    def predict(self, test_data, test_lables=None, singleton=False):
        """[summary]
        test_data ([type]): (向量长度，数据个数)
        test_lables ([type]): [description]
        """
        assert self.train_data is not None
        assert self.train_labels is not None
        if singleton:
            test_data = test_data.reshape(test_data.size, 1)

        results = []
        distances = []
        for i in range(test_data.shape[1]):
            # 计算距离，每个训练数据计算出 len(train) 个距离
            test_vector = test_data[:, i:i + 1]  # 列向量，(target_dim, 1)
            diff = test_vector - self.train_data
            dist = np.linalg.norm(diff, axis=0)
            # 距离从小到大的索引
            dist_idx = np.argsort(dist)
            # 取前k个，进行归类统计
            k_dist_idx = dist_idx[: self.K]
            count = np.zeros((len(self.category),))
            for idx in k_dist_idx:
                count[self.train_labels[idx]] += 1  # 对应类别 + 1
            results.append(np.argmax(count))
            distances.append(dist)
        self.distances = distances
        if test_lables is None:
            return np.array(results)
            # return ['subject' + str(r) for r in results]
        else:
            mis_result = test_lables - np.array(results)
            mis_count = np.count_nonzero(mis_result)
            mis_rate = mis_count / mis_result.size
            return np.array(results), mis_rate


class Reduction:
    @classmethod
    def pca(cls, features, target_dim):
        """
        param features: (向量长, 向量个数) 例：(45045, 75)
        return features: (target_dim, 向量个数) 例：（10, 75）
        """
        # 中心化，减去均值
        mean = np.mean(features, axis=1, keepdims=True)  # (45045, 1)
        phi = features - mean  # (45045, 75)
        # 计算协方差矩阵的特征值和特征向量，ATA和AAT的特征值相同，特征向量成比例
        cov = np.dot(phi.T, phi)  # ATA (75, 75)，减少运算量
        # 此处特征向量：（75, 1），简化计算
        lam, vec = np.linalg.eig(cov)  # vec (75, 75) 特征向量
        vec = np.dot(phi, vec)  # (45045, 75) 特征向量
        idx = np.argsort(-lam)  # 特征值从小到大排序
        idx = idx[: target_dim] # 选取前target_dim个特征向量

        vec = vec[:, idx]  # (origin_dim, target_dim) 选取特征值较大的特征向量
        vec = vec / np.sqrt(np.abs(lam[idx]))  # 归一化

        features = vec.T.dot(phi)  # (target_dim, 75)

        return features, vec, mean

    @classmethod
    def lda(cls, features, labels, t_dim=0):
        fs = features
        c = len(set(labels))  # 类别数
        if t_dim > c:
            t_dim = c-1
        k = len(labels) // c  # 每类个数
        # 计算15个类别的样本均值，shape: (len, 15)
        means = [np.mean(features[:, i*k: i*k+k], axis=1) for i in range(c)]
        means = np.array(means).T
        num = np.full((c,), k)
        mean = np.sum(num * means, axis=1, keepdims=True) / np.sum(num)
        sw = sum([(fs - m).dot((fs - m).T) for m in np.expand_dims(means.T, 2)])  # 类内散散布矩阵
        sb = (num * (mean - means)).dot((mean - means).T)  # 类间散布矩
        s, v = np.linalg.eig(np.linalg.inv(sw).dot(sb))
        idx = np.argsort(-s)
        v = v[:, idx[: t_dim+1]]
        return v.T.dot(features), v
        pass


def resize(image_vectors, scale=4):
    t = []
    scale = int(scale)
    for iv in image_vectors.T:
        iv.resize(231, 195)
        t_shape = (195 // scale, 231 // scale)
        iv = cv2.resize(iv, t_shape)
        t.append(iv.flatten())
    return np.array(t).T


if __name__ == '__main__':
    pass
    # N = [3, 4, 5, 6, 7]
    # K = [5, 10, 15]
    # repeat = 5
    # neighbour = 5
    # dataset, labels = read_all()
       
    # def eigenface():
    #     avg_mis_of_N = []
    #     for n in N:
    #         avg_mis = []
    #         for k in K:
    #             k_mis = []
    #             # 重复多次取平均值
    #             for _ in range(repeat):
    #                 # 随机选取训练数据
    #                 (train_data, train_label), (test_data, test_label) = random_select_data(dataset, labels, n)
    #                 # 数据降维
    #                 train_data, eigen, mean = Reduction.pca(train_data, k)
    #                 test_data = eigen.T.dot(test_data - mean)
    #                 # 使用knn分类器
    #                 knn = KNN(neighbour)
    #                 knn.fit(train_data, train_label)
    #                 res, mis = knn.predict(test_data, test_label)
    #                 k_mis.append(mis)
    #             avg_mis.append(sum(k_mis) / len(k_mis))
    #         avg_mis_of_N.append(avg_mis)
    #     return avg_mis_of_N

    # avg_mis_of_N = eigenface()