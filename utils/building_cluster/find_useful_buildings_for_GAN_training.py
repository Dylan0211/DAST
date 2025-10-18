# -*- coding: utf-8 -*-

"""
1. 选定一个target building和一个context转换的情景（context 1 -> 4）
2. 手上有的数据就是target building的context 1 以及其他楼的context 1 和 4 的数据
3. 要做的就是从其他楼的context 1 和 4 中筛选出能train出对于target building效果最好的GAN的数据
4. 目前能做的就是用target building的context 1 对其他楼的context 1 数据进行筛选（聚类 + 相关性）
"""
import pandas as pd
import numpy as np
import pickle

from fastdtw import fastdtw


def find_training_buildings(target_building_name, source_context, building_name_list, num_training_buildings):
    # load source raw_data
    with open('./tmp_data/building_tmp_data/{}_GAN_training_data_dict.pkl'.format(target_building_name), 'rb') as r:
        data_dict = pickle.load(r)
    context_data_dict = data_dict['context_data_dict']
    target_building_source_load_trace = context_data_dict[source_context]['df_context'].loc[:, 'nor_el_season']

    each_train_building_source_context_data_similarity_dict = {}
    for this_building_name in building_name_list:
        if this_building_name == target_building_name:
            continue

        with open('./tmp_data/building_tmp_data/{}_GAN_training_data_dict.pkl'.format(this_building_name), 'rb') as r:
            data_dict = pickle.load(r)
        context_data_dict = data_dict['context_data_dict']
        this_building_source_load_trace = context_data_dict[source_context]['df_context'].loc[:, 'nor_el_season']

        try:
            this_similarity = fastdtw(this_building_source_load_trace, target_building_source_load_trace)[0]
            each_train_building_source_context_data_similarity_dict.update({this_building_name: this_similarity})
        except:
            continue

    # consider the two most similar buildings as GAN training raw_data
    sorted_train_building_list = sorted(each_train_building_source_context_data_similarity_dict.items(), key=lambda item: item[1])
    final_train_building_list = [sorted_train_building_list[i][0] for i in range(num_training_buildings)]
    return final_train_building_list

