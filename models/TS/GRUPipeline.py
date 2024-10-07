import torch
import torch.nn as nn
from builtins import RuntimeError

from numpy import dtype
from torch.nn.utils.rnn import pad_sequence

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length, output_size=1, *, components=3, mean=0, scale=1):
        super(GRUNetwork, self).__init__()
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

        normalized_x = (x.squeeze() - self.mean) / self.scale

        k = torch.fft.rfft(normalized_x, dim=1)

        frequency_component = k.real  # Действительная часть
        x_decomposed = torch.cat((frequency_component.unsqueeze(-1), normalized_x.unsqueeze(-1)), dim=-1)
        x_decomposed = x_decomposed.permute(0, 2, 1).reshape(x_decomposed.shape[0], -1, series_length+self.components)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, h_n = self.gru(x_decomposed, h0)
        out_last = out[:, -1, :]

        outputs = []
        for _ in range(self.output_length):
            output = self.fc(out_last)
            outputs.append(output.unsqueeze(1))
            out_last = self.decoder(output.unsqueeze(1), h_n)[0].squeeze(1)

        return torch.cat(outputs, dim=1)*self.scale + self.mean