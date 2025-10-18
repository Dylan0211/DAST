from models.DAST.daily_profile.sax import SAX
from fastdtw import fastdtw
import pickle
import random
import numpy as np
import pandas as pd


num_of_augmentation_dict = {
    '2w': 6,
    '1m': 3,
    '3m': 1,
    '6m': 1
}


def generate_daily_profile(target_building_name, source_context, given_data_length):
    # load raw data
    with open(r'./tmp_data/building_tmp_data/clusters_and_motifs.pkl', 'rb') as r:
        tmp_dict = pickle.load(r)
    patterns_within_each_cluster = tmp_dict['clusters']
    motifs = tmp_dict['motifs']

    # daily profile augmentation
    df_synthetic = []
    for _ in range(num_of_augmentation_dict[given_data_length]):
        with open(r'./tmp_data/building_tmp_data/{}_season_{}_{}_data_dict.pkl'.format(target_building_name,source_context, given_data_length), 'rb') as r:
            data_dict = pickle.load(r)
        source_context_data = data_dict['source_context_data']
        df_source_context = data_dict['df_source_context']

        for i in range(source_context_data.shape[0]):
            this_day_load_trace = source_context_data[i]
            if np.sum(this_day_load_trace) == 0:
                continue
            distances_to_motifs = []
            for motif in motifs:
                distances_to_motifs.append(fastdtw(motif, this_day_load_trace)[0])
            cluster_of_this_day = distances_to_motifs.index(min(distances_to_motifs))

            # daily profile augmentation by SMOTE
            load_traces_within_this_cluster = patterns_within_each_cluster[cluster_of_this_day]
            random_index = random.randint(0, load_traces_within_this_cluster.shape[0] - 1)
            another_day_load_trace = load_traces_within_this_cluster[random_index]
            augmented_this_day_load_trace = this_day_load_trace + random.random() * (this_day_load_trace - another_day_load_trace)
            source_context_data[i] = augmented_this_day_load_trace

        augmented_load_trace = source_context_data.flatten()
        df_source_context['nor_el_season'] = augmented_load_trace
        df_synthetic.append(df_source_context)

    df_synthetic = pd.concat(df_synthetic, axis=0)
    return df_synthetic


# def generate_daily_profile(target_building_name, source_context, given_data_length, wordSize, alphabetSize):
#     # load raw data
#     with open(r'./tmp_data/building_tmp_data/motifs_and_traces.pkl', 'rb') as r:
#         sax_representation_dict = pickle.load(r)
#
#     # daily profile augmentation
#     sax_trans = SAX(wordSize=wordSize, alphabetSize=alphabetSize)
#     df_synthetic = []
#     for _ in range(num_of_augmentation_dict[given_data_length]):
#         with open(r'./tmp_data/building_tmp_data/{}_season_{}_{}_data_dict.pkl'.format(target_building_name,source_context, given_data_length), 'rb') as r:
#             data_dict = pickle.load(r)
#         source_context_data = data_dict['source_context_data']
#         df_source_context = data_dict['df_source_context']
#
#         for i in range(source_context_data.shape[0]):
#             this_day_load_trace = source_context_data[i]
#             if np.sum(this_day_load_trace) == 0:
#                 continue
#             this_day_load_trace_rep = sax_trans.to_letter_rep(this_day_load_trace)[0]
#
#             if this_day_load_trace_rep not in list(sax_representation_dict.keys()):
#                 continue
#             else:
#                 load_traces = sax_representation_dict[this_day_load_trace_rep]['ts']
#                 random.seed()
#                 rand_indices = random.sample(range(load_traces.shape[0]), 2)
#                 augmented_this_day_load_trace = np.mean([load_traces[j] for j in rand_indices], axis=0)
#                 source_context_data[i] = augmented_this_day_load_trace
#
#         augmented_load_trace = source_context_data.flatten()
#         df_source_context['nor_el_season'] = augmented_load_trace
#         df_synthetic.append(df_source_context)
#
#     df_synthetic = pd.concat(df_synthetic, axis=0)
#     return df_synthetic


def generate_daily_profile_without_decomposition(target_building_name, source_context, given_data_length, wordSize, alphabetSize):
    # load raw data
    with open(r'./tmp_data/building_tmp_data/{}_season_{}_{}_data_dict.pkl'.format(target_building_name, source_context, given_data_length), 'rb') as r:
        data_dict = pickle.load(r)
    source_context_data = data_dict['source_context_data']
    df_source_context = data_dict['df_source_context']

    with open(r'./tmp_data/building_tmp_data/motifs_and_traces.pkl', 'rb') as r:
        sax_representation_dict = pickle.load(r)

    # daily profile augmentation
    sax_trans = SAX(wordSize=wordSize, alphabetSize=alphabetSize)
    for i in range(source_context_data.shape[0]):
        this_day_load_trace = source_context_data[i]
        if np.sum(this_day_load_trace) == 0:
            continue
        this_day_load_trace_rep = sax_trans.to_letter_rep(this_day_load_trace)[0]

        if this_day_load_trace_rep not in list(sax_representation_dict.keys()):
            continue
        else:
            load_traces = sax_representation_dict[this_day_load_trace_rep]['ts']
            rand_indices = random.sample(range(load_traces.shape[0]), 2)
            augmented_this_day_load_trace = np.mean([load_traces[j] for j in rand_indices], axis=0)
            source_context_data[i] = augmented_this_day_load_trace

    augmented_load_trace = source_context_data.flatten()
    df_source_context['nor_el'] = augmented_load_trace

    return df_source_context

