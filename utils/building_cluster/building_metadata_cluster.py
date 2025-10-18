import pickle
import pandas as pd
import numpy as np
from config import building_name_list
from sklearn.cluster import SpectralClustering


categories_columns_name_list = [
    'site_id',
    'primaryspaceusage',
    'sub_primaryspaceusage',
    'yearbuilt',
]


def build_categories_columns(df):
    # discretize sqm, sqft, lat, lng, yearbuilt
    # df['sqm'] = pd.qcut(df['sqm'], q=4, labels=['low_sqm', 'medium_low_sqm', 'medium_high_sqm', 'high_sqm'])
    # df['sqft'] = pd.qcut(df['sqft'], q=4, labels=['low_sqft', 'medium_low_sqft', 'medium_high_sqft', 'high_sqft'])
    # df['lat'] = pd.qcut(df['lat'], q=4, labels=['low_lat', 'medium_low_lat', 'medium_high_lat', 'high_lat'])
    # df['lng'] = pd.qcut(df['lng'], q=4, labels=['low_lng', 'medium_low_lng', 'medium_high_lng', 'high_lng'])
    df['yearbuilt'] = pd.qcut(df['yearbuilt'], q=4, labels=['low_yearbuilt', 'medium_low_yearbuilt', 'medium_high_yearbuilt', 'high_yearbuilt'])
    return df


def build_matrix_by_metadata(metadata_task_list):
    tasks_num = metadata_task_list.__len__()
    d_meta = np.zeros((tasks_num, tasks_num))

    # stage II: Jaccard Distance
    for i, metadata_1 in enumerate(metadata_task_list):
        for j, metadata_2 in enumerate(metadata_task_list):
            diff_count = 0  # metadata 不一样的数量
            for ii in range(len(categories_columns_name_list)):
                diff_count += (metadata_1[ii] != metadata_2[ii])
            j_d = 1 - (len(categories_columns_name_list) - diff_count) / (len(categories_columns_name_list) + diff_count)
            d_meta[i, j] = j_d

    return d_meta


# def metadata_cluster():
if __name__ == '__main__':
    # 1. load raw_data
    df = pd.read_csv('../../raw_data/metadata/metadata.csv', usecols=['building_id'] + categories_columns_name_list)
    building_metadata_list = []
    for i in range(df.shape[0]):
        if df.loc[i, 'building_id'] in building_name_list:
            building_metadata_list.append(df.iloc[i])
    df = pd.DataFrame(np.vstack(building_metadata_list))
    df.columns = ['building_id'] + categories_columns_name_list

    each_season_mean_nor_el_dict = {
        'winter': [],
        'spring': [],
        'summer': [],
        'fall': [],
    }
    for building_name in building_name_list:
        with open('../building_tmp_data/{}_GAN_training_data_dict.pkl'.format(building_name), 'rb') as r:
            data_dict = pickle.load(r)
        context_data_dict = data_dict['context_data_dict']
        for context_id, context_name in enumerate(['winter', 'spring', 'summer', 'fall']):
            each_season_mean_nor_el_dict[context_name].append(context_data_dict[context_id + 1][1]['nor_el'].mean())

    df['win_spr'] = [each_season_mean_nor_el_dict['winter'][i] / each_season_mean_nor_el_dict['spring'][i]
                     for i in range(len(each_season_mean_nor_el_dict['winter']))]
    df['win_sum'] = [each_season_mean_nor_el_dict['winter'][i] / each_season_mean_nor_el_dict['summer'][i]
                     for i in range(len(each_season_mean_nor_el_dict['winter']))]
    df['win_fal'] = [each_season_mean_nor_el_dict['winter'][i] / each_season_mean_nor_el_dict['fall'][i]
                     for i in range(len(each_season_mean_nor_el_dict['winter']))]
    df['spr_sum'] = [each_season_mean_nor_el_dict['spring'][i] / each_season_mean_nor_el_dict['summer'][i]
                     for i in range(len(each_season_mean_nor_el_dict['winter']))]
    df['spr_fal'] = [each_season_mean_nor_el_dict['spring'][i] / each_season_mean_nor_el_dict['fall'][i]
                     for i in range(len(each_season_mean_nor_el_dict['winter']))]
    df['sum_fal'] = [each_season_mean_nor_el_dict['summer'][i] / each_season_mean_nor_el_dict['fall'][i]
                     for i in range(len(each_season_mean_nor_el_dict['winter']))]

    df.fillna(np.nan, inplace=True)

    df['yearbuilt'] = pd.qcut(df['yearbuilt'], q=4, labels=['yearbuilt_{}'.format(i) for i in range(4)])
    df['win_spr'] = pd.qcut(df['win_spr'], q=8, labels=['win_spr_{}'.format(i) for i in range(8)])
    df['win_sum'] = pd.qcut(df['win_sum'], q=8, labels=['win_sum_{}'.format(i) for i in range(8)])
    df['win_fal'] = pd.qcut(df['win_fal'], q=8, labels=['win_fal_{}'.format(i) for i in range(8)])
    df['spr_sum'] = pd.qcut(df['spr_sum'], q=8, labels=['spr_sum_{}'.format(i) for i in range(8)])
    df['spr_fal'] = pd.qcut(df['spr_fal'], q=8, labels=['spr_fal_{}'.format(i) for i in range(8)])
    df['sum_fal'] = pd.qcut(df['sum_fal'], q=8, labels=['sum_fal_{}'.format(i) for i in range(8)])


    # categories_columns_name_list = categories_columns_name_list + ['win_spr', 'win_sum', 'win_fal', 'spr_sum', 'spr_fal', 'sum_fal']
    # categories_columns_name_list = categories_columns_name_list + ['win_fal']
    categories_columns_name_list = ['win_spr', 'win_sum', 'win_fal', 'spr_sum', 'spr_fal', 'sum_fal']

    print('Done 1. load csv raw_data')

    # 2. build raw_data into tasks
    task_dict = {}
    metadata_task_list = []
    for metadata, df_small in df.groupby(categories_columns_name_list):
        task_dict.update({metadata: df_small})
        metadata_task_list.append(metadata)
    print('Done 2. build raw_data into %s tasks' % metadata_task_list.__len__())

    # 3. build matrix
    d_mate_matrix = build_matrix_by_metadata(metadata_task_list=metadata_task_list)
    print('Done 3. build matrix')

    # 4. meta-clustering modeling
    metadata_clustering_dict = {}
    n_clusters = 5
    clustering = SpectralClustering(n_clusters=n_clusters,
                                    assign_labels='discretize',
                                    random_state=0).fit(d_mate_matrix)
    labels = clustering.labels_
    for c in range(n_clusters):
        metadata_clustering_dict.update({c: np.where(labels == c)[0]})

    for cluster_id in metadata_clustering_dict.keys():
        tmp_building_name_list = []
        for task_id in metadata_clustering_dict[cluster_id]:
            for building_name in task_dict[metadata_task_list[task_id]]['building_id']:
                tmp_building_name_list.append(building_name)
        metadata_clustering_dict[cluster_id] = tmp_building_name_list

    print(categories_columns_name_list)
    for cluster, buildings in metadata_clustering_dict.items():
        print('{}, {}'.format(cluster, buildings))

    # # 4.1 save clustering raw_data result
    # with open('save/metadata_task_list.pkl', 'wb') as w:
    #     pickle.dump(metadata_task_list, w)
    # with open('save/metadata_clustering_dict.pkl', 'wb') as w:
    #     pickle.dump(metadata_clustering_dict, w)





