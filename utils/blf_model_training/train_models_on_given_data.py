from utils.blf_model import *
from torch.utils.data import DataLoader

import torch


def train_model(data_x, data_y, model_name, input_dim, hidden_dim, num_layers, batch_size, learning_rate, num_of_BLF_training_epochs, \
                sparse_output_dim, output_dim, enc_hid_dim, dec_hid_dim):
    num = data_x.shape[0] // batch_size
    if num > 1:
        data_x = data_x[:(num * batch_size)]
        data_y = data_y[:(num * batch_size)]

    # parameters
    train_dataset = TrainSet(data_x, data_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

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
    model = model_dict[model_structure_name]

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # start training
    for epoch in range(num_of_BLF_training_epochs):
        l_sum = 0.
        for (x, y) in train_dataloader:
            x = x.to(torch.float32)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l_sum += loss.item()

        print('    epoch {}: loss = {}'.format(epoch + 1, l_sum / len(train_dataloader)))

    # save model
    save_path = './tmp_data/trained_BLF_models/{}.pt'.format(model_name)
    torch.save(model.state_dict(), save_path)
    print('    model saved: {}'.format(save_path))
