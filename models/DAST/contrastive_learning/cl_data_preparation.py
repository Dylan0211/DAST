from models.DAST.contrastive_learning.deviation_operations import *
import pickle
import numpy as np


def generate_positive_and_negative_samples(target_building_name, source_context, given_data_length):
    with open(r'./tmp_data/building_tmp_data/{}_season_{}_{}_data_dict.pkl'.format(target_building_name, source_context, given_data_length), 'rb') as r:
        data_dict = pickle.load(r)
    df_source_context = data_dict['df_source_context']
    original_season = df_source_context['nor_el_season']
    original_remainder = df_source_context['nor_el_remainder']
    num_samples = 10
    augmented_remainder_list = [original_remainder.sample(frac=1).reset_index(drop=True)
                                for _ in range(num_samples)]

    # create real and synthetic traces
    real_load_trace = original_season + original_remainder
    synthetic_load_trace_list = [original_season + augmented_remainder_list[i]
                                 for i in range(len(augmented_remainder_list))]
    for i in range(len(synthetic_load_trace_list)):
        for j in range(synthetic_load_trace_list[i].shape[0]):
            if synthetic_load_trace_list[i][j] > 1:
                synthetic_load_trace_list[i][j] = 1
            elif synthetic_load_trace_list[i][j] < 0:
                synthetic_load_trace_list[i][j] = 0
    each_day_load_trace_with_deviation_trace = []
    each_day_load_trace_with_deviation_removed_trace = []
    each_day_load_trace_with_deviation_augmented_trace = []
    for i in range(synthetic_load_trace_list[0].shape[0] // 24):
        each_day_load_trace_with_deviation_trace.append(real_load_trace[24 * i: 24 * (i + 1)])
        each_day_load_trace_with_deviation_removed_trace.append(original_season[24 * i: 24 * (i + 1)])
    for i in range(len(synthetic_load_trace_list)):
        for j in range(synthetic_load_trace_list[i].shape[0] // 24):
            each_day_load_trace_with_deviation_augmented_trace.append(synthetic_load_trace_list[i][24 * j: 24 * (j + 1)])

    # create positive and negative samples
    cl_sample_a = []
    cl_sample_b = []
    cl_labels = []
    for i in range(len(each_day_load_trace_with_deviation_trace)):
        for j in range(num_samples):
            cl_sample_a.append(np.array(each_day_load_trace_with_deviation_removed_trace[i]))
            cl_sample_b.append(np.array(each_day_load_trace_with_deviation_trace[i]))
            cl_labels.append(0)

        for j in range(num_samples):
            cl_sample_a.append(np.array(each_day_load_trace_with_deviation_removed_trace[i]))
            cl_sample_b.append(np.array(each_day_load_trace_with_deviation_augmented_trace[num_samples * i + j]))
            cl_labels.append(1)

    cl_sample_a = np.array(cl_sample_a)
    cl_sample_b = np.array(cl_sample_b)
    cl_labels = np.array(cl_labels)

    return cl_sample_a, cl_sample_b, cl_labels


# def generate_positive_and_negative_samples(target_building_name, source_context, given_data_length):
#     with open(r'./tmp_data/building_tmp_data/{}_season_{}_{}_data_dict.pkl'.format(target_building_name, source_context, given_data_length), 'rb') as r:
#         data_dict = pickle.load(r)
#     df_source_context = data_dict['df_source_context']
#     original_season = df_source_context['nor_el_season']
#     original_remainder = df_source_context['nor_el_remainder']
#     augmented_remainder = original_remainder.sample(frac=1).reset_index(drop=True)
#
#     # create real and synthetic traces
#     real_load_trace = original_season + original_remainder
#     synthetic_load_trace = original_season + augmented_remainder
#     for i in range(synthetic_load_trace.shape[0]):
#         if synthetic_load_trace[i] > 1:
#             synthetic_load_trace[i] = 1
#         elif synthetic_load_trace[i] < 0:
#             synthetic_load_trace[i] = 0
#     each_day_load_trace_with_deviation_list = []
#     each_day_load_trace_with_deviation_removed_list = []
#     each_day_load_trace_with_deviation_augmented_list = []
#     for i in range(synthetic_load_trace.shape[0] // 24):
#         each_day_load_trace_with_deviation_list.append(real_load_trace[24 * i: 24 * (i + 1)])
#         each_day_load_trace_with_deviation_removed_list.append(original_season[24 * i: 24 * (i + 1)])
#         each_day_load_trace_with_deviation_augmented_list.append(synthetic_load_trace[24 * i: 24 * (i + 1)])
#
#     # create positive and negative samples
#     cl_samples = []
#     cl_labels = []
#     for i in range(len(each_day_load_trace_with_deviation_list)):
#         tmp_trace_1 = np.array(each_day_load_trace_with_deviation_removed_list[i])
#         tmp_trace_2 = np.array(each_day_load_trace_with_deviation_list[i])
#         this_sample = np.concatenate((tmp_trace_1, tmp_trace_2))
#         cl_samples.append(this_sample)
#         cl_labels.append(0)
#
#         tmp_trace_1 = np.array(each_day_load_trace_with_deviation_removed_list[i])
#         tmp_trace_2 = np.array(each_day_load_trace_with_deviation_augmented_list[i])
#         this_sample = np.concatenate((tmp_trace_1, tmp_trace_2))
#         cl_samples.append(this_sample)
#         cl_labels.append(1)
#
#     cl_samples = np.array(cl_samples)
#     cl_labels = np.array(cl_labels)
#
#     random_index = np.random.permutation(cl_samples.shape[0])
#     cl_samples = cl_samples[random_index]
#     cl_labels = cl_labels[random_index]
#
#     return cl_samples, cl_labels


if __name__ == '__main__':
    generate_positive_and_negative_samples()
