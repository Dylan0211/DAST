from config import *
from utils.source_data_preparation import dataloader_for_source_data
from utils.building_cluster.find_useful_buildings_for_GAN_training import find_training_buildings
from utils.blf_model_evaluation.test_model_on_original_data import test_model
from utils.blf_model_training.train_models_on_given_data import train_model
from utils.evaluation_metrics.MMD import mmd_rbf
from utils.evaluation_metrics.TSNE import draw_tsne_for_real_and_syn_data
from models.DAST.load_context_transformation.generate_data_using_GAN import generate_synthetic_data_using_GAN, generate_synthetic_temperature
from models.DAST.load_context_transformation.train_GAN_model import train_GAN_models_with_selected_buildings
from models.DAST.daily_profile.generate_daily_profile import generate_daily_profile
from models.DAST.remainder.remainder_augmentation import combine_load_traces_with_augmented_remainder
from models.DAST.contrastive_learning.cl_data_preparation import generate_positive_and_negative_samples
from models.DAST.contrastive_learning.cl_model_training import train_cl_model
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    synthetic_data = []
    df_synthetic = []
    print('*** Target Building: {}'.format(target_building_name))
    for target_context in all_context_set:
        if target_context in given_context_set:
            source_context = target_context
        else:
            possible_source_context_list = source_context_dict.get(target_context)
            source_context = None
            for i in range(len(possible_source_context_list)):
                if possible_source_context_list[i] in given_context_set:
                    source_context = possible_source_context_list[i]
                    break

        if source_context is None:
            continue

        print('*** Current Context: {}'.format(target_context))
        print('1. raw data preparation')
        print('    load source raw data')
        dataloader_for_source_data.read_data_for_context(target_building_name=target_building_name,
                                                         source_context=source_context,
                                                         seq_length=seq_length,
                                                         given_data_length=given_data_length,
                                                         daily_separation_point=daily_separation_point)

        if target_context in given_context_set:
            print('2. daily profile augmentation')
            df_synthetic_in_this_context = generate_daily_profile(target_building_name=target_building_name,
                                                                  source_context=source_context,
                                                                  given_data_length=given_data_length)

            print('3. train contrastive learning model')
            cl_sample_a, cl_sample_b, cl_label = generate_positive_and_negative_samples(target_building_name=target_building_name,
                                                                                        source_context=source_context,
                                                                                        given_data_length=given_data_length)
            cl_model = train_cl_model(sample_a=cl_sample_a, sample_b=cl_sample_b, labels=cl_label,
                                      batch_size=batch_size,
                                      num_epochs=num_of_cl_model_epochs, learning_rate=learning_rate)

            print('4. remainder augmentation')
            df_synthetic_in_this_context = combine_load_traces_with_augmented_remainder(df_synthetic_in_this_context=df_synthetic_in_this_context,
                                                                                        cl_model=cl_model, num_of_noises=num_of_noises)
            df_synthetic_in_this_context.reset_index(drop=True, inplace=True)

            synthetic_data_in_this_context = []
            for i in range(df_synthetic_in_this_context.shape[0] // seq_length):
                synthetic_data_in_this_context.append(np.array(df_synthetic_in_this_context.loc[i * seq_length: (i + 1) * seq_length - 1, 'nor_el']))
            synthetic_data_in_this_context = np.array(synthetic_data_in_this_context)

            synthetic_data.append(synthetic_data_in_this_context)
            df_synthetic.append(df_synthetic_in_this_context)
            print()
        else:
            print('2. task clustering')
            train_building_name_list = find_training_buildings(target_building_name=target_building_name,
                                                               source_context=source_context,
                                                               building_name_list=building_name_list,
                                                               num_training_buildings=num_training_buildings)
            print('    selected buildings for GAN training: {}'.format(train_building_name_list))

            print('3. GAN model training')
            print('    start GAN training')
            gen_a_save_path = './tmp_data/trained_UNIT_models/{}_season_{}_to_{}_gen.pt'.format(target_building_name,
                                                                                                source_context,
                                                                                                target_context)
            gen_b_save_path = './tmp_data/trained_UNIT_models/{}_season_{}_to_{}_gen.pt'.format(target_building_name,
                                                                                                target_context,
                                                                                                source_context)

            # note: 不重复train GAN
            if not os.path.exists(gen_a_save_path) or not os.path.exists(gen_b_save_path):
                train_GAN_models_with_selected_buildings(train_building_name_list=train_building_name_list,
                                                         source_context=source_context, target_context=target_context,
                                                         batch_size=batch_size, gen_learning_rate=gen_learning_rate,
                                                         dis_learning_rate=dis_learning_rate,
                                                         num_epochs=num_of_UNIT_training_epochs,
                                                         alpha=alpha, gen_a_save_path=gen_a_save_path,
                                                         gen_b_save_path=gen_b_save_path)
            else:
                print('    GAN model already exists')
            # note: 重复train GAN
            # train_GAN_models_with_selected_buildings(train_building_name_list=train_building_name_list,
            #                                          source_context=source_context, target_context=target_context,
            #                                          batch_size=batch_size, gen_learning_rate=gen_learning_rate,
            #                                          dis_learning_rate=dis_learning_rate,
            #                                          num_epochs=num_of_UNIT_training_epochs,
            #                                          alpha=alpha, gen_a_save_path=gen_a_save_path,
            #                                          gen_b_save_path=gen_b_save_path)
            print('    GAN model saved at: {}'.format(gen_a_save_path))
            print('    GAN model saved at: {}'.format(gen_b_save_path))
            print('    GAN training end')

            print('4. domain transformation')
            delta_temp = generate_synthetic_temperature(source_context=source_context,
                                                        target_context=target_context,
                                                        target_building_name=target_building_name)
            df_synthetic_in_this_context = generate_synthetic_data_using_GAN(target_building_name=target_building_name,
                                                                             gen_a_save_path=gen_a_save_path,
                                                                             gen_b_save_path=gen_b_save_path,
                                                                             batch_size=batch_size,
                                                                             delta_temp=delta_temp,
                                                                             source_context=source_context,
                                                                             given_data_length=given_data_length)

            print('5. train contrastive learning model')
            cl_sample_a, cl_sample_b, cl_label = generate_positive_and_negative_samples(target_building_name=target_building_name,
                                                                                        source_context=source_context,
                                                                                        given_data_length=given_data_length)
            cl_model = train_cl_model(sample_a=cl_sample_a, sample_b=cl_sample_b, labels=cl_label,
                                      batch_size=batch_size,
                                      num_epochs=num_of_cl_model_epochs, learning_rate=learning_rate)

            print('6. remainder augmentation')
            df_synthetic_in_this_context = combine_load_traces_with_augmented_remainder(df_synthetic_in_this_context=df_synthetic_in_this_context,
                                                                                        cl_model=cl_model, num_of_noises=num_of_noises)
            df_synthetic_in_this_context.reset_index(drop=True, inplace=True)

            synthetic_data_in_this_context = []
            for i in range(df_synthetic_in_this_context.shape[0] // seq_length):
                synthetic_data_in_this_context.append(np.array(df_synthetic_in_this_context.loc[i * seq_length: (i + 1) * seq_length - 1, 'nor_el']))
            synthetic_data_in_this_context = np.array(synthetic_data_in_this_context)

            synthetic_data.append(synthetic_data_in_this_context)
            df_synthetic.append(df_synthetic_in_this_context)
            print()

    synthetic_data = np.concatenate(synthetic_data, axis=0)
    df_synthetic = pd.concat(df_synthetic, axis=0)
    print('    synthetic_data shape: {}'.format(synthetic_data.shape))
    print('    df_synthetic shape: {}'.format(df_synthetic.shape))
    print()

    # evaluate quality of synthetic data
    print('start synthetic data evaluation')
    given_context = list(given_context_set)[0]
    with open('./tmp_data/building_tmp_data/{}_season_{}_{}_data_dict.pkl'.format(target_building_name, given_context,
                                                                                  given_data_length), 'rb') as r:
        data_dict = pickle.load(r)
    df = data_dict.get('df')

    df = df.loc[int(df.shape[0] * 0.5):]
    df.reset_index(drop=True, inplace=True)

    original_data = []
    for i in range(df.shape[0] // seq_length):
        original_data.append(np.array(df.loc[i * seq_length: (i + 1) * seq_length - 1, 'nor_el']))
    original_data = np.array(original_data)
    original_data = original_data[:synthetic_data.shape[0]]
    print('    original data shape: {}'.format(original_data.shape))

    # draw
    load = df_synthetic.loc[:, 'nor_el']
    with open('tmp_data/output_files/synthetic_load_trace.pkl', 'wb') as w:
        pickle.dump(load, w)
    plt.plot(range(load.shape[0]), load, color='blue')
    plt.show()

    # MMD outputs
    # mmd = mmd_rbf(source=original_data.flatten(), target=synthetic_data.flatten())
    # print('    MMD = {:.4f}'.format(mmd))

    # t-SNE outputs
    # # note: for t-SNE
    # syn_data = synthetic_data.copy()
    # real_data = original_data.copy()
    # print('    real data length: {}'.format(real_data.shape[0]))
    # print('    syn data length: {}'.format(syn_data.shape[0]))
    # draw_tsne_for_real_and_syn_data(real_data=real_data, syn_data=syn_data, title='fig_tsne_dast')
