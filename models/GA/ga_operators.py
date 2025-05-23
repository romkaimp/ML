import numpy as np
import torch
from typing import Optional

from sympy.stats.rv import probability

from models.GA.baseChromosome import Chromosome


class Ops:
    def __init__(self, comb_prob: float, mutate_prob: float):
        self.c_p = comb_prob
        self.m_p = mutate_prob

    @staticmethod
    def get_gens(chrom: Chromosome):
        cur_state = chrom.state_dict()
        filtered_values = list(filter(lambda item: item[0] in chrom.trainable_layers(), cur_state.items()))
        values = [item[1] for item in filtered_values]
        return values

    @staticmethod
    def nose(population: list):
        for i in range(len(population)):

            state: dict = population[i].state_dict()
            for name, layer in state.items():
                if name not in population[i].trainable_layers():
                    continue
                else:
                    layer += torch.tensor(np.random.normal(0, 1, size=layer.shape),
                                                       dtype=layer.dtype)

    @staticmethod
    def combinate_self(array: torch.Tensor, prob: float) -> torch.Tensor:
        device = array.device
        random_vals = torch.rand(array.shape, device=device)
        mask = random_vals < prob
        flat_array = array.view(-1)
        shuffle_indices = torch.arange(flat_array.size(0), device=device)[mask.view(-1)]
        shuffled_indices = shuffle_indices[torch.randperm(shuffle_indices.size(0), device=device)]
        flat_array[mask.view(-1)] = flat_array[shuffled_indices]
        return flat_array.view(array.shape)

    @staticmethod
    def mutate_self(array: torch.Tensor, prob: float) -> torch.Tensor:
        random_vals = torch.rand_like(array)
        mask = random_vals < prob

        noise = torch.randn(array.shape, dtype=array.dtype, device=array.device)

        array = array.clone()
        array[mask] += noise[mask]

        return array

    @staticmethod
    def combinate_parents(p1, p2, prob):
        probability = prob

        random_vals = np.random.rand(*p1.shape)
        mask = random_vals < probability
        flat_array1 = p1.flatten()
        flat_array2 = p2.flatten()
        shuffle_indices = np.where(mask.flatten())[0]
        np.random.shuffle(shuffle_indices)
        els = flat_array1[shuffle_indices]
        flat_array1[mask.flatten()] = flat_array2[shuffle_indices]
        flat_array2[mask.flatten()] = els
        return flat_array1.reshape(p1.shape), flat_array2.reshape(p1.shape)