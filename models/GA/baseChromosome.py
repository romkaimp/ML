import torch
from torch.nn import Module
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod
from models.TS.GRUPipeline import Predictor
import scipy

class Chromosome(ABC, Module):
    @abstractmethod
    def get_params(self):
        """возвращает параметры при инициализации"""
        pass

    @staticmethod
    def wb(layer_names: tuple) -> tuple:
        new_tuple = [(layer_names[i] + ".bias", layer_names[i] + ".weight") for i in range(len(layer_names))]
        new_tuple = [item for tup in new_tuple for item in tup]
        return tuple(new_tuple)

    @abstractmethod
    def trainable_layers(self):
        pass

class LinearChromosome(Chromosome):
    def __init__(self, input, output, model: Predictor, pred_length, hidden_size):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_length = pred_length
        self.model = model.to(self.device)
        self.input = input
        self.output = output

        self.conv = nn.Conv1d(10, 15, 3).to(self.device)
        self.lr1 = nn.Linear(20 - 3 + 1, hidden_size).to(self.device)
        self.activation = nn.ReLU()
        self.lr2 = nn.Linear(pred_length, hidden_size).to(self.device)
        self.last = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x):
        super().__init__()
        batch_size, series_number, series_length = x.size()
        x = x.to(self.device)

        self.model.eval()
        y = self.model(x).squeeze()
        y = self.lr2(y) #[batch_size, pred_length] -> [batch_size, hidden_size]

        x = self.lr1(x)
        x = self.activation(x)
        return self.lr2(x)

    @property
    def get_params(self):
        return self.input, self.output

    @property
    def trainable_layers(self) -> tuple:
        return self.wb(("lr1", "lr2",))