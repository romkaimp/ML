import torch
from torch.nn import Module
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod
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
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output

        self.lr1 = nn.Linear(input, input*10)
        self.activation = nn.ReLU()
        self.lr2 = nn.Linear(input*10, output)

    def forward(self, x):
        super().__init__()
        x = self.lr1(x)
        x = self.activation(x)
        return self.lr2(x)

    @property
    def get_params(self):
        return self.input, self.output

    @property
    def trainable_layers(self) -> tuple:
        return self.wb(("lr1", "lr2",))