from utils.blf_model import *
from torch.utils.data import DataLoader

import torch
import numpy as np
import math
import matplotlib.pyplot as plt


def draw_mse_and_rmse(label_list, pred_list, model_name):
    mae_list = [abs(label_list[i] - pred_list[i]) for i in range(len(pred_list))]
    mae = sum(mae_list) / len(mae_list)
    rmse_list = [(label_list[i] - pred_list[i]) ** 2 for i in range(len(pred_list))]
    rmse = math.sqrt(sum(rmse_list) / len(rmse_list))

    fig = plt.figure(figsize=(10, 6))
    fig.add_subplot(111)
    plt.plot(range(len(pred_list)), pred_list, marker='8', color='orange', linewidth=1, label='predict')
    plt.plot(range(len(label_list)), label_list, color='b', linewidth=1, label='label')
    title = 'MAE={:.3f} \n RMSE={:.3f}'.format(mae, rmse)
    plt.title(model_name)
    plt.title(title, loc='right')
    plt.legend()
    plt.show()


def test_models(data_x, data_y, load_max, load_min, model_name_list, input_dim, hidden_dim, num_layers, batch_size,
                sparse_output_dim, output_dim, enc_hid_dim, dec_hid_dim):
    num = data_x.shape[0] // batch_size
    if num > 1:
        data_x = data_x[:(num * batch_size)]
        data_y = data_y[:(num * batch_size)]

    # GPU info
    if_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if if_cuda else "cpu")

    # test all model_structure
    mae_dict = {}
    for model_name in model_name_list:
        # set test set and loader
        test_set = TrainSet(data_x, data_y)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # set up trained_BLF_models
        model_dict = {
            'lstm': LSTM(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers),
            'sparse_lstm': Sparse_LSTM(input_dim=input_dim,
                                       hidden_dim=hidden_dim,
                                       num_layers=num_layers,
                                       sparse_output_dim=sparse_output_dim,
                                       output_dim=output_dim),
            'sparse_ed': Seq2Seq(input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 num_layers=num_layers,
                                 sparse_output_dim=sparse_output_dim,
                                 output_dim=output_dim),
            'seq2seq_with_attention': Seq2Seq_with_attention(input_dim=input_dim,
                                                             output_dim=output_dim,
                                                             enc_hid_dim=enc_hid_dim,
                                                             dec_hid_dim=dec_hid_dim),
        }
        if model_name.startswith('lstm'):
            model_structure_name = 'lstm'
        elif model_name.startswith('sparse_lstm'):
            model_structure_name = 'sparse_lstm'
        elif model_name.startswith('sparse_ed'):
            model_structure_name = 'sparse_ed'
        elif model_name.startswith('seq2seq_with_attention'):
            model_structure_name = 'seq2seq_with_attention'
        model = model_dict.get(model_structure_name)

        # load state_dict
        save_path = './tmp_data/trained_BLF_models/{}.pt'.format(model_name)
        model.load_state_dict(torch.load(save_path))
        model = model.to(device)

        # start testing
        preds = []
        labels = []
        for x, y in test_loader:
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            x = x.to(device)

            pred = model(x)
            preds.append(pred.cpu().detach().numpy())
            labels.append(y.detach().numpy())

        # denormalize
        pred_list = []
        label_list = []
        for i in range(len(preds)):
            this_pred = preds[i].flatten()
            this_label = labels[i].flatten()
            for j in range(this_pred.shape[0]):
                pred_list.append(this_pred[j] * (load_max - load_min) + load_min)
                label_list.append(this_label[j] * (load_max - load_min) + load_min)

        # print inference results
        error_list = [abs(label_list[i] - pred_list[i]) for i in range(len(pred_list))]
        print('    model name: {}, MAE = {}'.format(model_name, np.array(error_list).mean()))

        mae_dict.update({model_name: np.array(error_list).mean()})

    sorted_model_list = list(dict(sorted(mae_dict.items(), key=lambda item: item[1])).keys())

    return sorted_model_list
