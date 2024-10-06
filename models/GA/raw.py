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
from models.GA.ga_operators import Ops

#trainer = Trainer()
class Trainer():
    def __init__(self, start_active):
        self.active = start_active
        self.cur = start_active
        self.bank = 0

    #data - R3 набор временных рядов подряд идущих разных временных отрезков
    def reward(self, data, model=None):
        rews = []
        #x - набор подряд идущих отрезков временных рядов R2
        for x in data:
            cur, bank = self.cur, self.bank
            #t - один временной ряд
            price = 0
            for t in x:
                y, amount = model(t)
                if y > 0.9:
                    c = cur
                    bank += min(amount, cur)/t[-1]
                    cur -= min(amount, c)
                elif y < 0.1:
                    b = bank
                    bank -= min(amount, bank)/t[-1]
                    cur += min(amount, b)
                price = t[-1]
            rews.append(cur + bank*price)
        return np.mean(rews)

    def count_rewards(self, data, population: list[Chromosome]):
        rewards = []
        for chromosome in population:
            rewards.append(self.reward(data, chromosome))
        return rewards


def choose_parents(population: list[Chromosome], target_values: tuple, size: int):
    size = min(size, len(population))

    probs = softmax(target_values)
    arr = [i for i in range(len(population))]
    new_population_idx = np.random.choice(arr, size, replace=False, p=probs)
    new_population = [population[i] for i in new_population_idx]
    return new_population

def combine_chromosomes(population: list[Chromosome], new_size, prob):
    new_population = []
    crossover = []
    for i in range(len(population)):
        crossover[i] = np.random.choice([j if j != i else (j**2)%len(population) for j in range(len(population))], 1)
    #Выбрали пары для каждой хромосомы
    #
    for i in range(len(population)):
        parent1: Chromosome = population[i]
        parent2 = population[crossover[i]]

        new_model1: Chromosome = LinearChromosome(*parent1.get_params)
        new_model2: Chromosome = LinearChromosome(*parent1.get_params)

        new_parent2: Chromosome = LinearChromosome(*parent1.get_params)
        new_parent1: Chromosome = LinearChromosome(*parent1.get_params)

        cur_state1: dict = copy.deepcopy(parent1.state_dict())
        cur_state2: dict = copy.deepcopy(parent2.state_dict())

        p_state1: dict = copy.deepcopy(parent1.state_dict())
        p_state2: dict = copy.deepcopy(parent2.state_dict())


        for name, layer1, _, layer2, _, p_layer1, _, p_layer2 in zip(cur_state1.items(), cur_state2.items(), parent1.state_dict().items(), parent2.state_dict().items()):
            if name not in parent1.trainable_layers:
                continue
            else:
                new_layer1, new_layer2 = Ops.combinate_parents(layer1, layer2, prob)
                cur_state1[name] = new_layer1
                cur_state2[name] = new_layer2

                p_state1[name] = Ops.combinate_self(p_layer1, prob)
                p_state2[name] = Ops.combinate_self(p_layer2, prob)

        new_model1.load_state_dict(cur_state1)
        new_model2.load_state_dict(cur_state2)
        new_parent1.load_state_dict(p_state1)
        new_parent2.load_state_dict(p_state2)

        new_population.append(new_model1)
        new_population.append(new_model2)
        new_population.append(new_parent1)
        new_population.append(new_parent2)

    if len(new_population) > new_size:
        idxs = np.random.choice([i for i in range(len(new_population))], new_size)
        pop = [new_population[idx] for idx in idxs]
        return pop
    elif len(new_population) < new_size:
        return combine_chromosomes(new_population, new_size)

def mutate_population(population: list[Chromosome], prob):
    for i in range(len(population)):

        state: dict = copy.deepcopy(population[i].state_dict())
        for name, layer in state.items():
            if name not in population[i].trainable_layers:
                continue
            else:
                Ops.combinate_self(layer, prob)

def check_convergence(rewards, eps) -> bool:
    rew = [sum(arr) for arr in rewards]
    if rew[-1] - rew[-2] < eps:
        return True
    else:
        return False

# train_data - массив из R3. 1dim - временной ряд, 2dim - набор временных рядов подряд идущих, достаточно продолжительных, 3dim - набор различных отрезков временных рядов
def cycle(start_population: list[Chromosome], train_data, n_max, eps, start_active, batch_size):
    trainer = Trainer(start_active)
    rewards = []
    rewards.append(-eps*3.3)
    rewards.append(-eps*2)
    epoch = 0
    # 1.  добавляем небольшой шум изначальным распределениям
    Ops.nose(start_population)
    while epoch < n_max:
        #2.  высчитываем награды
        idxs = np.random.choice([x for x in range(len(train_data))], min(batch_size, train_data.shape[0])) #  R3
        cur_data = [train_data[idx] for idx in idxs] #  R3
        cur_rewards = trainer.count_rewards(cur_data, start_population)
        rewards.append(sum(cur_rewards))

        #3. Проверяем, что награды отличаются
        if check_convergence(rewards, eps):
            break

        #4.  выбираем родителей генотипически
        parents = choose_parents(start_population, rewards[-1], 3)

        #5. Скрещиваем родительские гены
        population = combine_chromosomes(parents, 12, 0.1)

        #6. Мутируем родительские гены
        population = mutate_population(population, 0.1)

        start_population = population
        epoch += 1
