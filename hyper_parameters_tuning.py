import os
import numpy as np
import pandas as pd
import torch
from ray import tune
from functools import partial
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim


# Construção da Rede Neural
class LSTM(nn.Module):
    def __init__(self, nHLayers=1, l1=100, dropout=0.3):
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

    return normalizer.fit_transform(training_set)


def training_sliding_window(training_set):
    x_train = []
    y_train = []

    for i in range(7, training_set.shape[0]):
        x_train.append(training_set[i-7:i, 0])
        y_train.append(training_set[i, 0])

    return np.array(x_train), np.array(y_train)


def test_sliding_window(test_set):
    x_test = []
    for i in range(7, test_set.shape[0]):
        x_test.append(test_set[i-7:i, 0])
    return np.array(x_test)


# Treinamento
def training(config, checkpoint_dir=None):
    lstm = LSTM(config["nHLayers"], config["l1"], config["dropout"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            lstm = nn.DataParallel(lstm)
    lstm.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        lstm.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    normalizer = MinMaxScaler(feature_range=(0, 1))
    training_set = load_data(normalizer)

    x_train, y_train = training_sliding_window(training_set)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    data = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(data, batch_size=int(config["batch_size"]), shuffle=True)

    for epoch in range(int(config["epoch"])):
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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((lstm.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=running_loss)

        print(f'ÉPOCA {epoch+1} FINALIZADA: custo {running_loss}')
    print("Treinamento Finalizado!")


def main(num_samples=5, max_num_epochs=1000, gpus_per_trial=1):
    config = {
        "nHLayers": tune.choice([1, 2, 3]),
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "dropout": tune.loguniform(0.5, 0.1),
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "epoch": tune.choice(list(range(10, 1001)))
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2
    )

    reporter = CLIReporter(metric_columns=["loss", "training_iteration"])
    
    result = tune.run(
        partial(training),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print("Best trial final training loss: {}".format(best_trial.last_result["loss"]))


if __name__ == '__main__':
    main(num_samples=50, max_num_epochs=1000, gpus_per_trial=1)
