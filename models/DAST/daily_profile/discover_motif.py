import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from config import building_name_list, all_context_set
from models.DAST.daily_profile.kmeans_ts_clustering import clustering_methodology


def discover_motif_using_dtw():
    normalized_load_trace_units = []
    load_trace_units = []
    for building_name in building_name_list:
        with open(r'./tmp_data/building_tmp_data/{}_GAN_training_data_dict.pkl'.format(building_name), 'rb') as r:
            data_dict = pickle.load(r)
        context_data_dict = data_dict['context_data_dict']
        for context in all_context_set:
            normalized_load_trace_units.append(context_data_dict[context]['normalized_context_data'])
            load_trace_units.append(context_data_dict[context]['context_data'])
    normalized_load_trace_units = np.vstack(normalized_load_trace_units)
    load_trace_units = np.vstack(load_trace_units)

    # note: the number of clusters depend on the size of daily profile
    # num_cluster = int(0.01 * load_trace_units.shape[0])
    num_cluster = 16

    centroids, closest_centroids = clustering_methodology(daily_load_trace_list=normalized_load_trace_units, cluster_num=num_cluster)

    with open(r'./tmp_data/building_tmp_data/centroids.pkl', 'wb') as w:
        pickle.dump({'centroids': centroids, 'closest_centroids': closest_centroids}, w)

    with open(r'./tmp_data/building_tmp_data/centroids.pkl', 'rb') as r:
        tmp_dict = pickle.load(r)
    centroids = tmp_dict['centroids']
    closest_centroids = tmp_dict['closest_centroids']

    patterns_within_each_cluster = []
    for cluster_id in range(num_cluster):
        cluster_i_load = load_trace_units[closest_centroids == cluster_id]
        cluster_i_norm_load = normalized_load_trace_units[closest_centroids == cluster_id]
        center_load_trace = np.mean(cluster_i_norm_load, axis=0)
        patterns_within_each_cluster.append(cluster_i_load)

        # draw
        plt.figure(figsize=(8, 6))
        for i in range(cluster_i_norm_load.shape[0]):
            plt.plot(range(cluster_i_norm_load.shape[1]), cluster_i_norm_load[i, :], color='r', alpha=0.1)
        plt.plot(range(cluster_i_norm_load.shape[1]), center_load_trace, color='b')
        plt.title("num of sample in this cluster: {},   {}".format(cluster_i_norm_load.shape[0],
                                                                   cluster_i_norm_load.shape[0] / float(
                                                                       load_trace_units.shape[0])))
        plt.savefig('./tmp_data/output_files/motif_{}.jpg'.format(cluster_id))

    # save centroids and corresponding mean curve
    with open(r'./tmp_data/building_tmp_data/clusters_and_motifs.pkl', 'wb') as w:
        pickle.dump({'clusters': patterns_within_each_cluster, 'motifs': centroids}, w)
    print('centroids and motifs saved')
