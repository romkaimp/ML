import numpy as np
import torch
import torch.nn as nn
from builtins import RuntimeError
from models.TS.fracdiff.torch import fdiff
from models import SpectralConv1d

from numpy import dtype
from torch.nn.utils.rnn import pad_sequence
from models.SpectralConv1d import FNO1d
from abc import ABC, abstractmethod


class Predictor(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, x, times):
        pass


class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length, output_size=1, *, components=3, mean=0,
                 scale=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_length = output_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.components = components

        self.mean = mean
        self.scale = scale
        # x.shape = (b_size, input_size)
        # Настройка GRU слоя
        self.gru = nn.GRU(components + input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.decoder = nn.GRU(output_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        # Полносвязный слой для превратить выходные данные GRU в требуемый формат
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, series_number, series_length = x.size()

        # normalized_x = (x.squeeze() - self.mean) / self.scale

        k = torch.fft.rfft(x, dim=1)

        frequency_component = k.real  # Действительная часть
        x_decomposed = torch.cat((frequency_component.unsqueeze(-1), normalized_x.unsqueeze(-1)), dim=-1)
        x_decomposed = x_decomposed.permute(0, 2, 1).reshape(x_decomposed.shape[0], -1, series_length + self.components)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, h_n = self.gru(x_decomposed, h0)
        out_last = out[:, -1, :]

        outputs = []
        for _ in range(self.output_length):
            output = self.fc(out_last)
            outputs.append(output.unsqueeze(1))
            out_last = self.decoder(output.unsqueeze(1), h_n)[0].squeeze(1)

        return torch.cat(outputs, dim=1)  # *self.scale + self.mean


class GRUwithFNO(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, num_layers, output_length, output_size=1, *,
                 components=3, mean=0, scale=1):
        super(GRUwithFNO, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_length = output_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.components = components

        self.mean = mean
        self.scale = scale

        self.activation = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(sequence_length)
        self.batch_norm2 = nn.BatchNorm1d(3)

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.decoder = nn.GRU(output_size, hidden_size, num_layers, batch_first=True, dropout=0.1)

        self.fno = FNO1d(sequence_length, 3, components, 10, nn.ReLU(), 3)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_length)

    def forward(self, x):
        batch_size, series_number, series_length = x.size()

        normalized_x = (x)  # - self.mean) / self.scale
        normalized_x = self.batch_norm1(normalized_x)

        fno_x = self.fno(normalized_x)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, h_n = self.gru(fno_x, h0)

        output = out[:, -1, :]
        output = self.fc2(output)[:, :, np.newaxis]
        for _ in range(self.output_length):
            out_last, h_n = self.decoder(output, h_n)
            out_last = self.fc1(out_last.reshape(batch_size * self.output_length, self.hidden_size)).reshape(batch_size,
                                                                                                             self.output_length)[
                       :, :, np.newaxis]
            output = out_last

        return output


class GRUFrac(Predictor):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, *,
                 mean=0, scale=1, activation, mean_d=0, std_d=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        self.mean = mean
        self.scale = scale
        self.mean_d = mean_d
        self.scale_d = std_d

        self.d = 2
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2).to(
            self.device)
        self.frac_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,
                               dropout=0.2).to(self.device)
        self.diff_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True,
                               dropout=0.2).to(self.device)

        self.flat = nn.Flatten().to(self.device)
        self.fc = nn.Linear(hidden_size * 3 * self.d, hidden_size).to(self.device)

        self.batch_norm = nn.BatchNorm1d(num_features=hidden_size, momentum=0.9).to(self.device)
        self.last = nn.Linear(hidden_size, output_size).to(self.device)
        self.drop = nn.Dropout(p=0.5).to(self.device)

    def forward(self, x):
        batch_size, series_number, series_length = x.size()
        x = x.to(self.device)
        window = 10

        x_frac = fdiff(x.reshape(batch_size, series_number), dim=-1, n=0.6, window=window).reshape(batch_size,
                                                                                                   series_number, 1)
        x_frac = (((x_frac[:, window:, :] - torch.mean(x_frac[:, window:, :], dim=1)) / x_frac[:, window:, :].std())
        ).to(self.device)
        x_frac = torch.clamp(x_frac, min=-3, max=3)

        x_diff = torch.diff(x, n=1, dim=1)
        x_diff = (x_diff[:, 1:, :] / self.scale_d  # - x_diff[:, 1:, :].mean()
                  ).to(self.device)
        x_diff = torch.clamp(x_diff, min=self.mean_d - 3*self.scale_d, max=self.mean_d + 3*self.scale_d)

        normalized_x = (x - x.mean().item()
                        ) / x.std().item()
        normalized_x = torch.clamp(normalized_x, min=-3, max=+3)
        # normalized_x = x
        # plt.plot(x_frac[0].squeeze().detach().cpu(), color="red")
        # plt.plot(x_diff[0].squeeze().detach().cpu(), color="green")
        # plt.plot(normalized_x[0].squeeze().detach().cpu(), color="blue")
        # plt.show()
        # time.sleep(10)
        h0 = torch.zeros(self.d * self.num_layers, batch_size, self.hidden_size).to(self.device)

        out1, _ = self.gru(normalized_x, h0)
        out2, _ = self.frac_gru(x_frac, h0)
        out3, _ = self.diff_gru(x_diff, h0)

        output = torch.cat((out1[:, -1, :],
                            out2[:, -1, :], out3[:, -1, :]), dim=1).to(self.device)
        # output = self.flat(output)
        output = self.fc(output)
        # output = self.batch_norm(output)
        output = self.activation(output)
        output = self.drop(output) * x.std().item() + x.mean().item()
        output = self.last(output)
        return output

    def predict(self, x, times):
        batch_size, series_number, series_length = x.size()
        for i in range(times):
            ans = self.forward(x).reshape(batch_size, 1, 1)
            x = torch.cat((x, ans), dim=1)
        return x