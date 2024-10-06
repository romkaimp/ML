import torch
import torch.nn as nn
from PyEMD import CEEMDAN
from builtins import RuntimeError

from numpy import dtype
from torch.nn.utils.rnn import pad_sequence

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length, output_size=1, components=3):
        super(GRUNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_length = output_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.components = components
        # x.shape = (b_size, input_size)
        # Настройка GRU слоя
        self.gru = nn.GRU(components + input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.decoder = nn.GRU(output_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        # Полносвязный слой для превратить выходные данные GRU в требуемый формат
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, series_number, series_length = x.size()
        # CEEMDAN decomposition
        ceemdan = CEEMDAN(parallel=True)

        # Для каждого временного ряда в батче проводим EMD и разбиваем его на три дополнительных временных ряда
        decomposed_series = []
        for i in range(batch_size):
            imfs = ceemdan(x[i].cpu().numpy().squeeze())  # проводим EMD на временном ряде
            imfs = imfs[:self.components]  # берем только три первых IMFs
            imfs = torch.cat((torch.tensor(imfs, dtype=torch.float32), x[i].transpose(0, 1)), dim=0)
            # Добавляем -1 размер для консистенции
            decomposed_series.append(imfs.to(self.device))

        x_decomposed = pad_sequence(decomposed_series, batch_first=True)
        x_decomposed = x_decomposed.permute(0, 2, 1).reshape(x_decomposed.shape[0], -1, series_length+self.components)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, h_n = self.gru(x_decomposed, h0)
        out_last = out[:, -1, :]

        outputs = []
        for _ in range(self.output_length):
            output = self.fc(out_last)
            outputs.append(output.unsqueeze(1))
            out_last = self.decoder(output.unsqueeze(1), h_n)[0].squeeze(1)

        return torch.cat(outputs, dim=1)