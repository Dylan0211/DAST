from utils.source_data_preparation.dataloader_for_GAN_training_data import read_data
from utils.blf_model_training.train_models_on_given_data import train_model
from utils.source_data_preparation.dataloader_for_source_data import read_data_for_context
from models.DAST.daily_profile.discover_motif import discover_motif_using_dtw
from utils.blf_model import *
from config import *
import os
import pickle
import warnings
import numpy as np


warnings.filterwarnings('ignore')


def read_data_for_all_buildings():
    for building_name in building_name_list:
        # if os.path.exists(r'./tmp_data/building_tmp_data/{}_GAN_training_data_dict.pkl'.format(target_building_name)):
        #     continue
        read_data(target_building_name=building_name)


def train_blf_models_on_specified_seasons():
    for building_name in model_building_name_list:
        if not os.path.exists(r'./tmp_data/building_tmp_data/{}_season_1_{}_data_dict.pkl'.format(target_building_name, given_data_length)):
            read_data_for_context(target_building_name=building_name, source_context=1, seq_length=seq_length,
                                  given_data_length=given_data_length, daily_separation_point=daily_separation_point)
        for season_name in season_name_list:
            for model_structure_name in model_structure_name_list:
                model_name = '{}_{}_{}'.format(model_structure_name, season_name, building_name)
                if os.path.exists(r'./tmp_data/trained_BLF_models/{}.pt'.format(model_name)):
                    continue

                with open(r'./tmp_data/building_tmp_data/{}_season_1_{}_data_dict.pkl'.format(target_building_name, given_data_length), 'rb') as r:
                    data_dict = pickle.load(r)
                df = data_dict['df']

                # note: 只取前一年的数据来训练 BLF 模型
                df = df.loc[:int(0.5 * df.shape[0]), :]
                if season_name == 'win_spr':
                    df = df[df['season'] != 3]
                    df = df[df['season'] != 4]
                elif season_name == 'spr_sum':
                    df = df[df['season'] != 1]
                    df = df[df['season'] != 4]
                elif season_name == 'sum_fal':
                    df = df[df['season'] != 1]
                    df = df[df['season'] != 2]
                df.reset_index(inplace=True, drop=True)

                X, Y = [], []
                for i in range(df.shape[0] - 2 * seq_length):
                    X.append(np.array(df.iloc[i: i + seq_length, 6:-2]))
                    Y.append(np.array(df.loc[i + seq_length: i + 2 * seq_length - 1, 'nor_el']))
                X, Y = np.array(X), np.array(Y)

                train_model(data_x=X, data_y=Y, model_name=model_name, input_dim=input_dim, hidden_dim=hidden_dim,
                            num_layers=num_layers, batch_size=batch_size, learning_rate=learning_rate,
                            num_of_BLF_training_epochs=num_of_BLF_training_epochs, sparse_output_dim=sparse_output_dim,
                            output_dim=output_dim, enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim)


if __name__ == '__main__':
    # note: read building data for UNIT training
    read_data_for_all_buildings()

    # note: train BLF models
    # train_blf_models_on_specified_seasons()

    # note: discover motif   for all buildings
    # discover_motif_using_dtw()

