# from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.DAST.contrastive_learning.cl_data_preparation import generate_positive_and_negative_samples


class TrainSet(Dataset):
    def __init__(self, sample_a, sample_b, label):
        super(TrainSet, self).__init__()
        self.sample_a = sample_a
        self.sample_b = sample_b
        self.label = label

    def __getitem__(self, index):
        return self.sample_a[index], self.sample_b[index], self.label[index]

    def __len__(self):
        return len(self.label)


class ContrastiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ContrastiveNetwork, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x1, x2):
        x1, _ = self.encoder(x1)
        x2, _ = self.encoder(x2)
        distance = torch.square(x1 - x2)
        output = torch.sigmoid(self.fc(distance))
        return output


def train_cl_model(sample_a, sample_b, labels, batch_size, num_epochs, learning_rate):
    # raw_data processing
    separation_point = int(0.5 * labels.shape[0])
    train_sample_a, train_sample_b, train_label = sample_a[:separation_point], sample_b[:separation_point], labels[:separation_point]
    val_sample_a, val_sample_b, val_label = sample_a[separation_point:], sample_b[separation_point:], labels[separation_point:]

    train_set = TrainSet(train_sample_a, train_sample_b, train_label)
    val_set = TrainSet(val_sample_a, val_sample_b, val_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # GPU info
    if_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if if_cuda else "cpu")

    # model
    cl_model = ContrastiveNetwork(input_size=24, hidden_size=64).to(device)
    cl_criterion = nn.BCEWithLogitsLoss()
    cl_optimizer = torch.optim.Adam(params=cl_model.parameters(), lr=learning_rate)

    # train
    for epoch in range(num_epochs):
        loss = 0.0
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1 = x1.to(torch.float32)
            x2 = x2.to(torch.float32)
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.float().to(device)

            cl_optimizer.zero_grad()

            outputs = cl_model(x1, x2)
            loss = cl_criterion(outputs, y.unsqueeze(1))
            loss.backward()
            cl_optimizer.step()

            loss += loss.item()

        print('    Epoch {}, Loss: {:.4f}'.format(epoch + 1, loss/len(train_loader)))

    # validate
    cl_model.eval()
    total_cnt = 0
    match_cnt = 0
    with torch.no_grad():
        for x1, x2, y in val_loader:
            x1 = x1.to(torch.float32)
            x2 = x2.to(torch.float32)
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.float().to(device)

            outputs = cl_model(x1, x2)
            preds = []
            for output in outputs:
                if output > 0.5:
                    preds.append(1)
                else:
                    preds.append(0)

            for i in range(len(preds)):
                if preds[i] == y[i]:
                    match_cnt += 1
                total_cnt += 1
    print('    Validation accuracy: {}/{}'.format(match_cnt, total_cnt))


