import copy

import numpy as np
import torch
import torch.nn.utils as utils
from torch.nn import Module
import torch.nn as nn
from typing import Optional, Callable
from copy import deepcopy
from scipy.special import softmax
from models.GA.baseChromosome import Chromosome, LinearChromosome
from models.GA.ga_operators import Operators

class Selector():
    def __init__(self, comb, mutate):
        self.ops = Operators(comb, mutate)

    def combine_genes_(self, population: list[Chromosome]) -> None:
        for chromosome in population:
            self.ops.combinate_self( self.ops.get_gens(chromosome))

    def combine_chromosomes(self, population: list[Chromosome], chose: list[int]) -> list[Chromosome]:
        new_chromosomes = []
        for i, chromosome in enumerate(population):
            i_gene = copy.deepcopy(self.ops.get_gens(chromosome))
            i_gene_partner = copy.deepcopy( self.ops.get_gens(
                population[chose[i]]
            ))
            for gen1, gen2 in zip(i_gene, i_gene_partner):
                new_chromosome = self.ops.combinate_parents(gen1, gen2)




    def choose_parents(self, population: list[Chromosome], target_values: tuple, size: int):
        probs = softmax(target_values)
        arr = [i for i in range(len(population))]
        new_population_idx = np.random.choice(arr, size, replace=False, p=probs)
        new_population = [population[i] for i in new_population_idx]
        return new_population

    def new_population(self, population: list[Chromosome], size: int):
        new_population = []
        crossover = []
        for i in range(len(population)):
            crossover[i] = np.random.choice([j for j in range(len(population))], 1)
        while len(new_population) < size:





if __name__ == "__main__":
    sel = Selector(0.1, 0.1)
    print(sel.choose_parents([0, 1, 2], [0, 1, 2], 2))
