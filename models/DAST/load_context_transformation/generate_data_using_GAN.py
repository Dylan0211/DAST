import pandas as pd

from models.DAST.UNIT_model import *
from torch.utils.data import DataLoader

import pickle
import numpy as np


num_of_augmentation_dict = {
    '2w': 6,
    '1m': 3,
    '3m': 1,
    '6m': 1
}


def generate_data(data_a, gen_a_save_path, gen_b_save_path, batch_size):
    # set up trained_BLF_models
    gen_a = Generator()
    gen_b = Generator()
    gen_a.load_state_dict(torch.load(gen_a_save_path))
    gen_b.load_state_dict(torch.load(gen_b_save_path))

    test_set = TrainSet(data_a, data_a)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)

    # start testing
    fake = []
    for x_a, _ in test_loader:
        x_a = x_a.to(torch.float32)
        content, _ = gen_a.encode(x_a)
        output = gen_b.decode(content)
        fake.append(output.detach().numpy())

    # raw_data processing
    fake_data = []
    for i in range(len(fake)):
        for j in range(fake[i].shape[0]):
            fake_data.append(fake[i][j])

    final_fake_data = []
    for i in range(len(fake_data)):
        for j in range(fake_data[i].shape[0]):
            final_fake_data.append(fake_data[i][j])

    return np.array(final_fake_data)


def generate_synthetic_temperature(source_context, target_context, target_building_name):
    with open(r'./tmp_data/building_tmp_data/{}_GAN_training_data_dict.pkl'.format(target_building_name), 'rb') as r:
        data_dict = pickle.load(r)
    season_avg_temp_dict = data_dict['season_avg_temp_dict']
    delta_temp = season_avg_temp_dict[target_context] - season_avg_temp_dict[source_context]

    return delta_temp


def generate_synthetic_data_using_GAN(target_building_name, gen_a_save_path, gen_b_save_path,
                                      batch_size, delta_temp, source_context, given_data_length):
    df_synthetic_total = []
    for _ in range(num_of_augmentation_dict[given_data_length]):
        # load source raw_data
        with open('./tmp_data/building_tmp_data/{}_season_{}_{}_data_dict.pkl'.format(target_building_name, source_context, given_data_length), 'rb') as r:
            data_dict = pickle.load(r)
        source_context_data = data_dict['source_context_data']
        df_source_context = data_dict['df_source_context']
        temperature_max = data_dict['temperature_max']
        temperature_min = data_dict['temperature_min']

        synthetic_nor_el_season_list = generate_data(data_a=source_context_data,
                                                     gen_a_save_path=gen_a_save_path,
                                                     gen_b_save_path=gen_b_save_path,
                                                     batch_size=batch_size)

        # create new df
        df_synthetic = df_source_context.copy()
        df_synthetic.loc[:, 'nor_el_season'] = synthetic_nor_el_season_list

        # modify temperature
        df_synthetic['nor_temp'] = (df_synthetic['temperature'] + delta_temp - temperature_min) / (temperature_max - temperature_min)
        df_synthetic_total.append(df_synthetic)

    df_synthetic_total = pd.concat(df_synthetic_total, axis=0)
    return df_synthetic_total


def generate_synthetic_data_using_GAN_without_decomposition(target_building_name, gen_a_save_path, gen_b_save_path,
                                                            batch_size, delta_temp, source_context, given_data_length):
    # load source raw_data
    with open('./tmp_data/building_tmp_data/{}_season_{}_{}_data_dict.pkl'.format(target_building_name, source_context, given_data_length), 'rb') as r:
        data_dict = pickle.load(r)
    source_context_data = data_dict['source_context_data']
    df_source_context = data_dict['df_source_context']
    temperature_max = data_dict['temperature_max']
    temperature_min = data_dict['temperature_min']

    synthetic_nor_el_season_list = generate_data(data_a=source_context_data,
                                          gen_a_save_path=gen_a_save_path,
                                          gen_b_save_path=gen_b_save_path,
                                          batch_size=batch_size)

    # note: test
    # import matplotlib.pyplot as plt
    # plt.plot(range(len(synthetic_nor_el_season_list)), synthetic_nor_el_season_list, color='r')
    # plt.savefig('x.png')

    # create new df
    df_synthetic = df_source_context.copy()
    df_synthetic.loc[:, 'nor_el'] = synthetic_nor_el_season_list

    # modify temperature
    df_synthetic['nor_temp'] = (df_synthetic['temperature'] + delta_temp - temperature_min) / (temperature_max - temperature_min)

    return df_synthetic

