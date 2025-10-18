import pandas as pd
import numpy as np


# note: choose the model structure for training (lstm, sparse_lstm, sparse_ed, seq2seq_with_attention)
chosen_model_structure = 'sparse_lstm'

# note: modify these parameters
target_building_name = 'Fox_education_Virginia'
num_training_buildings = 1

# note: get buildings
df = pd.read_csv(r'./raw_data/meters/cleaned/electricity_cleaned.csv')
selected_site_list = ['Panther', 'Fox', 'Rat', 'Bear', 'Bull', 'Peacock', 'Cockatoo']
selected_category_list = ['education', 'public', 'office']
building_name_list = []
for building_name in list(df.columns):
    if building_name.split('_')[0] in selected_site_list and building_name.split('_')[1] in selected_category_list:
        load_trace = df.loc[:, building_name]
        nan_count = np.isnan(load_trace).sum()
        nan_ratio = nan_count / load_trace.shape[0]
        if nan_ratio <= 0.03:
            building_name_list.append(building_name)

# note: starting point of each building
daily_separation_point = 2

# note: raw_data length = ['2w', '1m', '3m', '6m']
given_data_length = '3m'

# note: given context set is a subset of {1 (winter), 2 (spring), 3 (summer), 4 (fall)}
given_context_set = {1}
all_context_set = {1, 2, 3, 4}
remainder_context_set = all_context_set.difference(given_context_set)

# note: number of noises inserted (range from 0 to 24)
num_of_noises = 12

'----------------------------------------------------------------------------------------------------------------------'
# key = 要生成的 context, val = 用于生成该 context 的 source context
source_context_dict = {
    1: [1, 2, 4, 3],
    2: [2, 1, 3, 4],
    3: [3, 1, 2, 4],
    4: [4, 1, 2, 3],
}

# BLF models
model_building_name_list = building_name_list[:15]
model_structure_name_list = ['lstm', 'sparse_lstm']
season_name_list = ['win_spr']
model_name_list = ['{}_{}_{}'.format(model_structure_name, season_name, building_name)
                   for model_structure_name in model_structure_name_list
                   for season_name in season_name_list
                   for building_name in model_building_name_list]

# epochs
num_of_UNIT_training_epochs = 50
num_of_CVAE_training_epochs = 50
num_of_TimeGAN_training_epochs = 50
num_of_RCGAN_training_epochs = 50
num_of_BLF_training_epochs = 10
num_of_cl_model_epochs = 10
temperature_max = 40
temperature_min = 0
seq_length = 24
batch_size = 64
batch_size_in_buffer = 24 * 7
gen_learning_rate = 1e-3
dis_learning_rate = 1e-4
learning_rate = 1e-3
alpha = 0.5

# BLF model parameters
input_dim = 33
sparse_output_dim = 64
hidden_dim = 128
output_dim = 1
num_layers = 1
enc_hid_dim = 128
dec_hid_dim = 128
