# -*- coding: utf-8 -*-

"""

@file: another_example.py
@time: 2023/4/30 19:40
@desc: 跟在SAX representation后面，得到cluster center, 拥有样本多的cluster的center即为motif

"""
import numpy as np
import random
from matplotlib import pyplot as plt


def cal_CHScore(num_samples, num_clusters, centroids, overall_centroid, daily_sax_represent_list, centroid_index_list,
                sax_trans):
    cohesion = 0.
    for i in range(num_clusters):
        cluster_i = daily_sax_represent_list[centroid_index_list == i]
        for j in range(cluster_i.shape[0]):
            cohesion += sax_trans.compare_strings(centroids[i], cluster_i[j])
    separation = 0.
    for i in range(num_clusters):
        num_samples_within_cluster_i = daily_sax_represent_list[centroid_index_list == i].shape[0]
        separation += num_samples_within_cluster_i * sax_trans.compare_strings(centroids[i], overall_centroid)

    return (separation / (num_clusters - 1)) / (cohesion / (num_samples - num_clusters))


def cal_silhouette_coefficient(num_samples, daily_sax_represent_list, centroid_index_list, sax_trans):
    s = []
    for i in range(num_samples):
        # a: 同蔟平均距离
        a = np.mean([sax_trans.compare_strings(daily_sax_represent_list[j], daily_sax_represent_list[i])
                     for j in range(num_samples)
                     if centroid_index_list[j] == centroid_index_list[i]])
        # b: 最近不同蔟平均距离
        b = min([np.mean([sax_trans.compare_strings(daily_sax_represent_list[j], daily_sax_represent_list[i])
                          for j in range(num_samples)
                          if centroid_index_list[j] != centroid_index_list[i]
                          and centroid_index_list[j] == k])
                 for k in set(centroid_index_list)
                 if k != centroid_index_list[i]])
        # s: 轮廓系数
        s_i = (b - a) / max(a, b) if (a != 0 or b != 0) else 0.0
        s.append(s_i)

    return np.mean(s)


def cal_mean_symbolic(symbolic_list, sax_trans):
    """
    求mean
    :param symbolic_list: ['a', 'b', 'b', 'd', 'a']
    :return: distance总和最小的那个字母
    """
    # the_letter_list = ['a', 'b', 'c', 'd', 'e']  # todo 字母集 ---- 应该根据 参数A 自动生成字母集  --- github 151行
    number_rep = range(0, sax_trans.get_alphabet_size())
    the_letter_list = [chr(x + sax_trans.get_aOffset()) for x in number_rep]

    distances = np.zeros(the_letter_list.__len__())

    for i in range(the_letter_list.__len__()):
        for item in symbolic_list:
            distances[i] += sax_trans.compare_letters(la=the_letter_list[i], lb=item)

    return the_letter_list[np.argmin(distances)]  # todo two letters may be both min


def clustering_methodology(daily_sax_represent_list, cluster_num, sax_trans):
    """
    用的k-means的思路
    :param cluster_num:
    :return: new centers, 以及各cluster的样本的索引 [0,1,0,1,1,0,1,1] -- if cluster_num=2
    """

    # daily_sax_represent_list = [
    #     'abc',  # len= 12
    #     'abd',
    #     'cdf',
    #     'abd'
    # ]  # todo this is loaded from SAX module

    daily_sax_represent_list = np.array(daily_sax_represent_list)

    # sax_trans = SAX_trans(ts, w, alpha)

    # # 随机初始化k个中心点
    sax_repre_set = set(daily_sax_represent_list)
    centroids = random.sample(sax_repre_set, cluster_num)

    CHScore = 0.
    silhouette_coefficient = 0.
    while True:
        # 1 计算每个数据点到所有中心点的距离
        distances = np.zeros((cluster_num, len(daily_sax_represent_list)))
        for i in range(cluster_num):
            for j in range(len(daily_sax_represent_list)):
                distances[i, j] = sax_trans.compare_strings(centroids[i], daily_sax_represent_list[j])

        # 2 找出距离每个中心点最近的数据点的索引
        closest_centroids = np.argmin(distances, axis=0)

        # 3 对于每个中心点，更新其位置为所有距离它最近的数据点的平均值
        new_centroids = []  # size is cluster num
        for i in range(cluster_num):
            cluster_i = daily_sax_represent_list[closest_centroids == i]
            the_mean_string = ""  # 对于这个cluster，重新计算出一个center, e.g., "abda"
            for k in range(daily_sax_represent_list[0].__len__()):  # string length, e.g., 12, 8
                # 按顺序，对每个位置的符号求一次“mean”
                point_list = [cluster_i[i][k] for i in range(cluster_i.shape[0])]
                the_mean_letter = cal_mean_symbolic(symbolic_list=point_list,
                                                    sax_trans=sax_trans)  # 找到那个平均值 'b' ---对于51个a， 49个c；在某个index k
                the_mean_string += the_mean_letter
            new_centroids.append(the_mean_string)

        overall_centroid = ""
        for i in range(daily_sax_represent_list[0].__len__()):
            point_list = [daily_sax_represent_list[j][i] for j in range(daily_sax_represent_list.shape[0])]
            the_mean_letter = cal_mean_symbolic(symbolic_list=point_list, sax_trans=sax_trans)
            overall_centroid += the_mean_letter

        # 计算轮廓系数
        # last_silhouette_coefficient = silhouette_coefficient
        # silhouette_coefficient = cal_silhouette_coefficient(num_samples=daily_sax_represent_list.shape[0],
        #                                                     daily_sax_represent_list=daily_sax_represent_list,
        #                                                     centroid_index_list=closest_centroids,
        #                                                     sax_trans=sax_trans)
        # print('[silhouette coefficient]', silhouette_coefficient)
        #
        # if silhouette_coefficient - last_silhouette_coefficient <= 1:
        #     break
        # centroids = new_centroids

        # 用CHScore来判断收敛效果如何
        last_CHScore = CHScore
        CHScore = cal_CHScore(num_samples=daily_sax_represent_list.shape[0],
                              num_clusters=cluster_num,
                              centroids=centroids,
                              overall_centroid=overall_centroid,
                              daily_sax_represent_list=daily_sax_represent_list,
                              centroid_index_list=closest_centroids,
                              sax_trans=sax_trans)
        print('[CHScore]:', CHScore)

        if CHScore - last_CHScore <= 0.5:
            break
        centroids = new_centroids

        # 4 如果新旧中心点之间的距离非常小，则认为算法已经收敛并退出循环
        # centre_distance = 0
        # for i in range(cluster_num):
        #     centre_distance += sax_trans.compare_strings(sA=new_centroids[i], sB=centroids[i])
        # print('[distance]:', centre_distance)
        #
        # if centre_distance < 1e-6:
        #     break
        # centroids = new_centroids

    return centroids, closest_centroids, silhouette_coefficient
