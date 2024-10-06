import numpy as np
import torch
from sympy.stats.rv import probability
import torch.nn.utils as utils
from torch.nn import Module
import torch.nn as nn
from typing import Optional, Callable
from models.GA.baseChromosome import Chromosome, LinearChromosome
from models.GA.ga_operators import Operators

#TODO поменять NN на абстрактный класс
#TODO добавить кроссингover родителей
class Genetics(Module):
    def __init__(self, cls: Callable, population, n_select, BASE_WINNERS: Optional[list[Module]], comb, mutate):
        """cls - нейросеть для обучения генетическими алгоритмами,
        layers - set(), из имён слоёв для оптимизации,
        population - количество копий хромосом в одной популяции,
        n_select - количество копий которые в дальнейшем будут копироваться,
        BASE_WINNER - нейросети из которых будут копироваться первые серии"""

        super().__init__()
        self.population = population
        self.n_select = n_select
        self.BASE_WINNERS = BASE_WINNERS
        self.model = cls
        self.selector = Selection(comb, mutate)

    def choose_gens(self, gens: list[Module], rewards: np.ndarray[float]) -> list[Module]:
        winners = []
        for i in range(self.n_select):
            idx = np.argmax(rewards)
            winners.append(gens[idx])
        return winners

    @staticmethod
    def divide_particles(n, m):
        base_count = n // m
        remainder = n % m
        result = [base_count + (1 if i < remainder else 0) for i in range(m)]

        return result

    def spawn_over_gens(self, gens: list[Chromosome]):
        ns = self.divide_particles(self.population, len(gens))
        new_gens: list[Module] = []
        for i, gen in enumerate(gens):
            for j in range(ns[i]):
                new_gens.append(self.model(gen.get_params()))

    def combinate_mutate(self, model: Chromosome, n) -> list[Chromosome]:
        """Дана изначальная модель нейросети, надо скопировать её (комбинировать + мутировать) n раз.
        Для этого выполянем действия:
        копируем модель с параметрами get_params
        преобразуем веса модели с shuffle_..., mutate_...
        сохраняем с использованием load_state."""
        new_models = []
        params = model.get_params()
        layers: set = model.get_layers()

        for i in n:
            new_model: Chromosome = self.model(params)
            cur_state: dict = model.state_dict()

            for name, layer in model.state_dict().items():
                if name not in layers:
                    continue
                else:
                    new_layer = self.shuffle_with_probability(layer)
                    new_layer = self.mutate_with_probability(new_layer)
                    cur_state[name] = new_layer

            new_model.load_state_dict(cur_state)
            new_models.append(new_model)

        return new_models

    def ME_diff(self):
        """считает ожидаемую прибавку к весам модели"""

    def converge(self, last_models: list[Chromosome], winners: list[Chromosome]):
        n_winners = len(winners)
        n_prev_models = len(last_models)
        for winner in winners:
            pass
