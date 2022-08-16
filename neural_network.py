import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler


# Construção da Rede Neural
class LSTM(nn.Module):
    def __init__(self, nHLayers, l1, dropout):
        super(LSTM, self).__init__()
        self.nHLayers = nHLayers
        self.l1 = l1
        self.l2 = int(l1 / 2)
        self.drop = dropout

        if self.nHLayers == 3:
            self.rnn_1 = nn.LSTM(input_size=1, hidden_size=self.l1)
            self.rnn_2 = nn.LSTM(self.l1, self.l2)
            self.rnn_3 = nn.LSTM(self.l2, self.l2, dropout=self.drop, num_layers=2)
            self.linear = nn.Linear(in_features=self.l2, out_features=1)
        elif self.nHLayers == 2:
            self.rnn_1 = nn.LSTM(input_size=1, hidden_size=self.l1)
            self.rnn_2 = nn.LSTM(self.l1, self.l2)
            self.linear = nn.Linear(in_features=self.l2, out_features=1)
        else:
            self.rnn_1 = nn.LSTM(input_size=1, hidden_size=self.l1)
            self.linear = nn.Linear(in_features=self.l1, out_features=1)

        self.dropout = nn.Dropout(p=self.drop)

    def forward(self, x):
        x = x.permute(1, 0).unsqueeze(2)

        if self.nHLayers == 3:
            x, _ = self.rnn_1(x)
            x = self.dropout(x)

            x, _ = self.rnn_2(x)
            x = self.dropout(x)

            x, _ = self.rnn_3(x)
        elif self.nHLayers == 2:
            x, _ = self.rnn_1(x)
            x = self.dropout(x)

            x, _ = self.rnn_2(x)
            x = self.dropout(x)
        else:
            x, _ = self.rnn_1(x)
            x = self.dropout(x)

        x = x[-1]
        x = self.dropout(x)
        x = self.linear(x)

        return x


# Carregamento e Tratamento da Base de Dados
def load_data(normalizer):
    training_data = pd.read_csv('/content/drive/MyDrive/dataset/training_data.csv')
    training_set = training_data.iloc[:, 1:2].values

    test_data = pd.read_csv('/content/drive/MyDrive/dataset/test_data.csv')
    real_values = test_data.iloc[:, 1:2].values

    complete_data = pd.concat((training_data['Vazões'], test_data['Vazões']), axis=0)

    test_set = complete_data[len(complete_data) - len(test_data) - 7:].values
    test_set = test_set.reshape(-1, 1)

    return normalizer.fit_transform(training_set), normalizer.fit_transform(test_set), real_values


def training_sliding_window(training_set):
    x_train = []
    y_train = []

    for i in range(7, training_set.shape[0]):
        x_train.append(training_set[i - 7:i, 0])
        y_train.append(training_set[i, 0])

    return np.array(x_train), np.array(y_train)


def test_sliding_window(test_set):
    x_test = []
    for i in range(7, test_set.shape[0]):
        x_test.append(test_set[i - 7:i, 0])
    return np.array(x_test)


# Treinamento
def training(lstm, epochs, loader, optimizer, criterion, device):
    errors = []
    for epoch in range(epochs):
        running_loss = 0.

        for i, data in enumerate(loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = lstm(inputs)
            outputs = outputs.flatten()

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        running_loss /= len(loader)
        errors.append(running_loss)
        print(f'ÉPOCA {epoch + 1} FINALIZADA: custo {running_loss}')
    return errors


# Previsão
def prediction(lstm, x_test, real_values, errors, normalizer):
    lstm.eval()
    predictions = lstm.forward(x_test)

    predictions = predictions.detach().cpu().numpy().reshape(-1, 1)

    predictions = normalizer.inverse_transform(predictions)

    print(predictions.mean())

    print(real_values.mean())

    errors = np.array(errors)
    plt.figure(figsize=(18, 6))
    graph_errors = plt.subplot(1, 2, 1)
    graph_errors.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    graph_test = plt.subplot(1, 2, 2)
    graph_test.set_title('Tests')
    plt.plot(real_values, color='red', label='Valor real')
    plt.plot(predictions, color='blue', label='Previsões')
    plt.xlabel('Dias')
    plt.ylabel('Vazão')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    normalizer = MinMaxScaler(feature_range=(0, 1))
    training_set, test_set, real_values = load_data(normalizer)

    x_train, y_train = training_sliding_window(training_set)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_test = test_sliding_window(test_set)

    data = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)

    device = torch.device('cpu')

    x_test = torch.tensor(x_test, device=device, dtype=torch.float32)

    lstm = LSTM(nHLayers=3, l1=256, dropout=0.22848248544632915)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=0.0021918207261538646)
    lstm.to(device)

    errors = training(lstm, 855, loader, optimizer, criterion, device)

    prediction(lstm, x_test, real_values, errors, normalizer)
