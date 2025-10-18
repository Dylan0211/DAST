"""
实验 1：
target building是 LIH
context转换场景是 ashrae 1 to 4
a: OIE
b: 除开LIH外的所有building
c: 由evaluation metrics筛选出来的最好的几个building
aaa
"""

from models.DAST.UNIT_model import Generator, Discriminator, TrainSet
from torch.utils.data import DataLoader
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import pickle
import torch
import numpy as np


# def check_training_result(gen_a, gen_b, epoch, train_building_name_list, source_context, target_context):
#     """
#     Check generated raw_data and best-match MAE after several epochs
#     :param epoch
#     :param gen_a
#     :param gen_b
#     :return: best-match MAE
#     """
#     mae_list = []
#     for index, building_name in enumerate(train_building_name_list):
#         with open('building_tmp_data/{}_ashrae_{}_to_{}_data_dict.pkl'.format(building_name, source_context, target_context),
#                   'rb') as r:
#             save_dict = pickle.load(r)
#
#         # get domain a and domain b raw_data
#         context_a_data = save_dict.get('context_a_data')
#         context_b_data = save_dict.get('context_b_data')
#         load_max = save_dict.get('load_max')
#         load_min = save_dict.get('load_min')
#
#         data_a = []
#         data_b = []
#         if not context_a_is_weekend and not context_b_is_weekend:  # weekday to weekday
#             temp_a = min([len(context_a_data[i]) for i in range(5)])
#             temp_b = min([len(context_b_data[i]) for i in range(5)])
#             num_days = min(temp_a, temp_b)
#             for i in range(num_days):
#                 for j in range(5):
#                     data_a.append(context_a_data[j][i])
#                     data_b.append(context_b_data[j][i])
#         elif not context_a_is_weekend and context_b_is_weekend:  # weekday to weekend
#             temp_a = min([len(context_a_data[i]) for i in range(2)])  # 周一周二 -> 周六周日
#             temp_b = min([len(context_b_data[i]) for i in range(2)])
#             num_days = min(temp_a, temp_b)
#             for i in range(num_days):
#                 for j in range(2):
#                     data_a.append(context_a_data[j][i])
#                     data_b.append(context_b_data[j][i])
#         elif context_a_is_weekend and not context_b_is_weekend:  # weekend to weekday
#             temp_a = min([len(context_a_data[i]) for i in range(5)])
#             temp_b = min([len(context_b_data[i]) for i in range(5)])
#             num_days = min(temp_a, temp_b)
#             for i in range(num_days):
#                 for j in range(5):
#                     data_a.append(context_a_data[j][i])
#                     data_b.append(context_b_data[j][i])
#         else:  # weekend to weekend
#             temp_a = min([len(context_a_data[i]) for i in range(2)])
#             temp_b = min([len(context_b_data[i]) for i in range(2)])
#             num_days = min(temp_a, temp_b)
#             for i in range(num_days):
#                 for j in range(2):
#                     data_a.append(context_a_data[j][i])
#                     data_b.append(context_b_data[j][i])
#         data_a = np.array(data_a)
#         data_b = np.array(data_b)
#
#         test_dataset = TrainSet(data_a, data_b)
#         test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
#         print('test on: {}'.format(building_name))
#         print('data_a shape: {}'.format(data_a.shape))
#         print('data_b shape: {}'.format(data_b.shape))
#
#         # start testing
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         fake = []
#         for x_a, _ in test_loader:
#             x_a = x_a.to(torch.float32)
#             x_a = x_a.to(device)
#             content, _ = gen_a.encode(x_a)
#             output = gen_b.decode(content)
#             fake.append(output.cpu().detach().numpy())
#
#         # raw_data processing
#         fake_data = []
#         for i in range(len(fake)):
#             for j in range(fake[i].shape[0]):
#                 fake_data.append(fake[i][j])
#
#         # denormalize
#         denorm_fake_data = []
#         denorm_real_data = []
#         for i in range(len(fake_data)):
#             denorm_fake_data.append([fake_data[i][j] * (load_max - load_min) + load_min
#                                      for j in range(fake_data[i].shape[0])])
#             denorm_real_data.append([data_b[i][j] * (load_max - load_min) + load_min
#                                      for j in range(data_b[i].shape[0])])
#
#         # error
#         mae_list.append(cal_mae_for_best_match(generated_data=denorm_fake_data,
#                                                target_data=denorm_real_data,
#                                                is_weekend_flag=context_a_is_weekend))
#
#         # check raw_data generated
#         flattened_fake_data = []
#         for i in range(len(denorm_fake_data)):
#             for j in range(len(denorm_fake_data[i])):
#                 flattened_fake_data.append(denorm_fake_data[i][j])
#
#         fig = plt.figure(figsize=(10, 6))
#         fig.add_subplot(111)
#         plt.plot(range(len(flattened_fake_data)), flattened_fake_data, color='b')
#         plt.title("generated raw_data (Epoch: {})".format(epoch))
#         plt.savefig("{}/train_set_c_epoch_{}.png".format(train_result_save_path, epoch))


def train_GAN_models_with_selected_buildings(train_building_name_list, source_context, target_context, batch_size,
                                             gen_learning_rate, dis_learning_rate, num_epochs, alpha, gen_a_save_path,
                                             gen_b_save_path):
    # GPU info
    if_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if if_cuda else "cpu")
    print("    set program to device:", torch.cuda.get_device_name(device))

    source_el_list = []
    target_el_list = []
    for building_name in train_building_name_list:
        with open('./tmp_data/building_tmp_data/{}_GAN_training_data_dict.pkl'.format(building_name), 'rb') as r:
            data_dict = pickle.load(r)
        load_mapping_dict = data_dict['load_mapping_dict']
        this_source_el = load_mapping_dict[(source_context, target_context)][0]
        this_target_el = load_mapping_dict[(source_context, target_context)][1]
        source_el_list.append(this_source_el)
        target_el_list.append(this_target_el)
    data_a = np.concatenate(source_el_list)
    data_b = np.concatenate(target_el_list)

    # initialize trained_BLF_models
    gen_a = Generator().to(device)
    gen_b = Generator().to(device)
    dis_a = Discriminator().to(device)
    dis_b = Discriminator().to(device)

    # set up raw_data loader
    train_dataset = TrainSet(data_a, data_b)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    print('    data_a shape: {}'.format(data_a.shape))
    print('    data_b shape: {}'.format(data_b.shape))

    # set up parameters
    gen_params = list(gen_a.parameters()) + list(gen_b.parameters())
    dis_params = list(dis_a.parameters()) + list(dis_b.parameters())
    gen_opt = torch.optim.Adam(gen_params, lr=gen_learning_rate)
    dis_opt = torch.optim.Adam(dis_params, lr=dis_learning_rate)

    # start training
    for epoch in range(num_epochs):
        gen_loss_sum = 0
        dis_loss_sum = 0
        for i, (x_a, x_b) in enumerate(train_loader):
            x_a = x_a.to(torch.float32)
            x_a = x_a.to(device)
            x_b = x_b.to(torch.float32)
            x_b = x_b.to(device)

            # train discriminator
            dis_opt.zero_grad()
            # encode
            h_a, n_a = gen_a.encode(x_a)
            h_a, n_a = h_a.to(device), n_a.to(device)
            h_b, n_b = gen_b.encode(x_b)
            h_b, n_b = h_b.to(device), n_b.to(device)
            # decode (cross domain)
            x_ba = gen_a.decode(h_b + n_b)
            x_ba = x_ba.to(device)
            x_ab = gen_b.decode(h_a + n_a)
            x_ab = x_ab.to(device)
            # discriminator loss
            loss_dis_a = dis_a.cal_dis_loss(x_ba, x_a)
            loss_dis_b = dis_b.cal_dis_loss(x_ab, x_b)
            loss_dis_all = loss_dis_a + loss_dis_b
            dis_loss_sum += loss_dis_all
            loss_dis_all.backward()
            dis_opt.step()

            # train generator
            gen_opt.zero_grad()
            # encode
            h_a, n_a = gen_a.encode(x_a)
            h_a, n_a = h_a.to(device), n_a.to(device)
            h_b, n_b = gen_b.encode(x_b)
            h_b, n_b = h_b.to(device), n_b.to(device)
            # decode (within domain)
            x_a_recon = gen_a.decode(h_a + n_a)
            x_b_recon = gen_b.decode(h_b + n_b)
            # decode (cross domain)
            x_ba = gen_a.decode(h_b + n_b)
            x_ba = x_ba.to(device)
            x_ab = gen_b.decode(h_a + n_a)
            x_ab = x_ab.to(device)
            # encode again
            h_b_recon, n_b_recon = gen_a.encode(x_ba)
            h_b_recon, n_b_recon = h_b_recon.to(device), n_b_recon.to(device)
            h_a_recon, n_a_recon = gen_b.encode(x_ab)
            h_a_recon, n_a_recon = h_a_recon.to(device), n_a_recon.to(device)
            # decode again
            x_bab = gen_b.decode(h_b_recon + n_b_recon)
            x_aba = gen_a.decode(h_a_recon + n_a_recon)
            # generator loss
            # reconstruction loss
            loss_gen_recon_x_a = (1 - alpha) * torch.mean(torch.abs(x_a_recon - x_a)) \
                                 + alpha * fastdtw(x_a_recon.cpu().detach().numpy(),
                                                   x_a.cpu().detach().numpy(),
                                                   dist=euclidean)[0]
            loss_gen_recon_x_b = (1 - alpha) * torch.mean(torch.abs(x_b_recon - x_b)) \
                                 + alpha * fastdtw(x_b_recon.cpu().detach().numpy(),
                                                   x_b.cpu().detach().numpy(),
                                                   dist=euclidean)[0]
            # GAN loss
            loss_gen_adv_a = dis_a.cal_gen_loss(x_ba)
            loss_gen_adv_b = dis_b.cal_gen_loss(x_ab)
            # cycle-consistency loss
            loss_gen_cycle_cons_a = torch.mean(torch.abs(x_aba - x_a))
            loss_gen_cycle_cons_b = torch.mean(torch.abs(x_bab - x_b))
            # total loss
            loss_gen_all = loss_gen_recon_x_a + loss_gen_recon_x_b + \
                           loss_gen_adv_a + loss_gen_adv_b + \
                           loss_gen_cycle_cons_a + loss_gen_cycle_cons_b
            gen_loss_sum += loss_gen_all
            loss_gen_all.backward()
            gen_opt.step()

        print('    Epoch {}: Generator loss: {} \t Discriminator loss: {}'.format(epoch + 1, gen_loss_sum, dis_loss_sum))

    torch.save(gen_a.state_dict(), gen_a_save_path)
    torch.save(gen_b.state_dict(), gen_b_save_path)


if __name__ == '__main__':
    train_GAN_models_with_selected_buildings(train_building_name_list=['OIE', 'OXH'],
                                             source_context=2,
                                             target_context=5,
                                             batch_size=64,
                                             gen_learning_rate=1e-3,
                                             dis_learning_rate=1e-4,
                                             num_epochs=50,
                                             alpha=0.5,
                                             gen_a_save_path='../trained_UNIT_models/LIH_gen_a.pt',
                                             gen_b_save_path='../trained_UNIT_models/LIH_gen_b.pt')
