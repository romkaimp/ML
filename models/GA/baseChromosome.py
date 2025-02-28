import torch
from torch.nn import Module
import torch.nn as nn
from typing import Optional, Any
from abc import ABC, abstractmethod
import copy
from typing_extensions import override

from models.TS.GRUPipeline import Predictor
import scipy

class Chromosome(ABC, Module):
    @property
    @abstractmethod
    def get_params(self):
        """возвращает параметры при инициализации"""
        pass

    @staticmethod
    def wb(layer_names: tuple) -> tuple:
        new_tuple = [(layer_names[i] + ".bias", layer_names[i] + ".weight") for i in range(len(layer_names))]
        new_tuple = [item for tup in new_tuple for item in tup]
        return tuple(new_tuple)

    def trainable_layers(self):
        names = []
        for name, _ in self.state_dict().items():
            names.append(name)
        return tuple(names)

class LinearChromosome(Chromosome):
    def __init__(self, input_size, hidden_size: int, pred_model: Predictor, pred_length):
        super(LinearChromosome, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_model = pred_model
        self.pred_model = self.pred_model.to(self.device)
        self.pred_length = pred_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_size2 = max(hidden_size - 10, 10)
        self.num_layers = 2

        self.activation_h = nn.Sigmoid()
        self.activation_x = nn.ReLU()

        self.d = 2
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          batch_first=True, bidirectional=True, dropout=0.2)

        self.lr1 = LayerBlock(self.d * self.num_layers * self.hidden_size * 2, self.hidden_size2)

        self.lr2 = nn.Linear(self.hidden_size2, 1).to(self.device)

        self.lr_x_first = LayerBlock(self.hidden_size * self.d, self.hidden_size2)
        self.lr_x_last = LayerBlock(self.hidden_size * self.d, self.hidden_size2)

        self.lr2_x = nn.Linear(self.hidden_size2 * 2, 1)

    def forward(self, x):
        batch_size, series_number, series_length = x.size()
        h_0 = torch.zeros(self.d * self.num_layers, batch_size, self.hidden_size).to(self.device)
        x = torch.clamp(x, x.mean() - x.std()*3, x.mean() + x.std()*3)

        x_first, h_k = self.gru(x, h_0)

        x = self.pred_model.predict(x, self.pred_length).to(self.device)
        x_last, h_n = self.gru(x, h_0)

        h = torch.cat((h_k, h_n), dim=-1).transpose(0, 1).reshape(batch_size, -1).to(self.device)

        h = self.lr1(h)
        p = self.activation_h(self.lr2(h))

        x_first, x_last = x_first[:, -1, :].reshape(batch_size, -1), x_last[:, -1, :].reshape(batch_size, -1)
        x1 = self.lr_x_first(x_first)
        x2 = self.lr_x_last(x_last)
        x_res = self.activation_x(self.lr2_x(torch.cat((x1, x2), dim=-1)))

        return p, x_res

    @property
    def get_params(self):
        return self.input_size, self.hidden_size, copy.deepcopy(self.pred_model), self.pred_length

    @override
    def trainable_layers(self):
        names = []
        for name, params in self.state_dict().items():
            if (not name.startswith('pred_model.') and 'batch_norm' not in name
                    and 'activation' not in name and 'dropout' not in name):
                names.append(name)
        return tuple(names)


class LayerBlock(Chromosome):
    def __init__(self, input_size, output_size, dropout_rate=0.5):
        super(LayerBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(input_size, output_size)

        self.batch_norm = nn.BatchNorm1d(output_size)

        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    @property
    def get_params(self):
        return self.input_size, self.output_size, self.dropout_rate