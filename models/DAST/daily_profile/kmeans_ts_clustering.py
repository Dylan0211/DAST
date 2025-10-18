import numpy as np
import random
from fastdtw import fastdtw
from matplotlib import pyplot as plt


def cal_CHScore(num_samples, num_clusters, centroids, overall_centroid, daily_load_trace_list, centroid_index_list):
    cohesion = 0.
    for i in range(num_clusters):
        cluster_i = daily_load_trace_list[centroid_index_list == i]
        for j in range(cluster_i.shape[0]):
            cohesion += fastdtw(centroids[i], cluster_i[j])[0]
    separation = 0.
    for i in range(num_clusters):
        num_samples_within_cluster_i = daily_load_trace_list[centroid_index_list == i].shape[0]
        separation += num_samples_within_cluster_i * fastdtw(centroids[i], overall_centroid)[0]

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


def clustering_methodology(daily_load_trace_list, cluster_num):
    """
    用的k-means的思路
    :param cluster_num:
    :return: new centers, 以及各cluster的样本的索引 [0,1,0,1,1,0,1,1] -- if cluster_num=2
    """
    daily_load_trace_list = np.array(daily_load_trace_list)

    # 随机初始化k个中心点
    centroids = np.random.permutation(daily_load_trace_list)[:cluster_num, :]

    CHScore = 0.
    iter = 0
    while True:
        # 1 计算每个数据点到所有中心点的距离
        distances = np.zeros((cluster_num, len(daily_load_trace_list)))
        for i in range(cluster_num):
            for j in range(len(daily_load_trace_list)):
                distances[i, j] = fastdtw(centroids[i], daily_load_trace_list[j])[0]

        # 2 找出距离每个中心点最近的数据点的索引
        closest_centroids = np.argmin(distances, axis=0)

        # 3 对于每个中心点，更新其位置为所有距离它最近的数据点的平均值
        new_centroids = []  # size is cluster num
        for i in range(cluster_num):
            try:
                cluster_i = daily_load_trace_list[closest_centroids == i]
                if cluster_i.shape[0] == 0:
                    new_centroids.append(centroids[i])
                else:
                    new_centroids.append(np.mean(cluster_i, axis=0))
            except:
                cluster_i = daily_load_trace_list[closest_centroids == i]
                print(cluster_i)
        new_centroids = np.array(new_centroids)

        overall_centroid = 0.
        for i in range(daily_load_trace_list.shape[0]):
            overall_centroid += daily_load_trace_list[i]
        overall_centroid = overall_centroid / daily_load_trace_list.shape[0]

        # 用CHScore来判断收敛效果如何
        # last_CHScore = CHScore
        # CHScore = cal_CHScore(num_samples=daily_load_trace_list.shape[0],
        #                       num_clusters=cluster_num,
        #                       centroids=centroids,
        #                       overall_centroid=overall_centroid,
        #                       daily_load_trace_list=daily_load_trace_list,
        #                       centroid_index_list=closest_centroids)
        # print('[CHScore]:', CHScore)
        #
        # if CHScore - last_CHScore <= 10:
        #     break
        # centroids = new_centroids

        # 如果新旧中心点之间的距离非常小，则认为算法已经收敛并退出循环
        centre_distance = 0
        for i in range(cluster_num):
            centre_distance += fastdtw(new_centroids[i], centroids[i])[0]
        print('[distance]:', centre_distance)

        iter += 1
        if centre_distance < 1 or iter >= 5:
            break
        centroids = new_centroids

    return centroids, closest_centroids
