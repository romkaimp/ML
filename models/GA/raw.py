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
from torch.utils.data import DataLoader, TensorDataset

#trainer = Trainer()
class Trainer:
    #data - R3 набор временных рядов подряд идущих разных временных отрезков
    def reward(self, data: torch.Tensor, model: Chromosome=None):
        rews = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #data - один батч
        #x - набор подряд идущих отрезков временных рядов R2 [[..., ...], [.., ...]]
        for x in data:
            batches, length = x.size()
            x_model = x.reshape(batches, length, 1)
            price = x[-1, -1].item()
            y, amount = model(x_model) #[[], [], [], [], []], [[], [], [], [], []] batch_size, 1
            y, amount = y.squeeze(), amount.squeeze()

            bank = torch.tensor(0, device=device).float()
            account = torch.tensor(0, device=device).float()

            #amount - всегда количество покупаемой/продаваемой валюты
            #y - "уверенность" модели в покупке/продаже
            condition1 = y > 0.85
            account += torch.sum(amount[condition1].float())
            bank -= torch.sum(x[condition1, -1].float() * amount[condition1].float())

            condition2 = y < 0.15
            account -= torch.sum(amount[condition2].float())
            bank += torch.sum(x[condition2, -1].float() * amount[condition2].float())

            rews.append((bank + account*price).item())

        rews_tensor = torch.tensor(rews, device=device)
        return rews_tensor.mean().item()

    def count_rewards(self, data: torch.Tensor, population: list[Chromosome]):
        rewards = []
        for chromosome in population:
            rewards.append(self.reward(data, chromosome))
        return tuple(rewards)


def choose_parents(population: list[Chromosome], target_values: tuple, size: int):
    size = min(size, len(population))

    probs = softmax(target_values)
    arr = [i for i in range(len(population))]
    new_population_idx = np.random.choice(arr, size, replace=False, p=probs)
    new_population = [population[i] for i in new_population_idx]
    return new_population

def combine_chromosomes(population: list[Chromosome], new_size, prob=0.3):
    new_population = []
    crossover = []
    for i in range(len(population)):
        crossover.append(np.random.choice([j if j != i else (j**2)%len(population) for j in range(len(population))], 1)[0])
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
        print("i:", i)
        for (name, layer1), (_, layer2), (_, p_layer1), (_, p_layer2) in zip(cur_state1.items(), cur_state2.items(), parent1.state_dict().items(), parent2.state_dict().items()):
            if name not in parent1.trainable_layers():
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
    else:
        return new_population

def mutate_population(population: list[Chromosome], prob) -> None:
    for i in range(len(population)):

        state: dict = copy.deepcopy(population[i].state_dict())
        for name, layer in state.items():
            if name not in population[i].trainable_layers():
                continue
            else:
                Ops.combinate_self(layer, prob)


def check_convergence(rewards: list, eps: float) -> bool:
    if abs(rewards[-1] - rewards[-2]) < eps:
        return True
    else:
        return False

# train_data - массив из R3. 1dim - временной ряд, 2dim - набор временных рядов подряд идущих, достаточно продолжительных, 3dim - набор различных отрезков временных рядов
def cycle(start_population: list[Chromosome], train_data: DataLoader,
          n_max=20, eps=0.01, parents_number=3, new_pop=12, prob=0.2):
    trainer = Trainer()
    best_model = deepcopy(start_population[0])
    rewards = [-3.4 * eps, -2 * eps]
    epoch = 0
    # 1.  добавляем небольшой шум изначальным распределениям (опционально, т.к. торч это делает)
    Ops.nose(start_population)
    flag = False
    patience = 3
    patience_counter = 0
    while epoch < n_max or rewards[-1] < 0:
        if flag:
            break
        for i, data in enumerate(train_data):
            print("step:", i)
            data = data[0]
            #2.  высчитываем награды
            cur_rewards = trainer.count_rewards(data, start_population)
            rewards.append(sum(cur_rewards))
            if rewards[-1] > rewards[-2]:
                max_reward_index = cur_rewards.index(max(cur_rewards))
                best_model = copy.deepcopy(start_population[max_reward_index])

            #3. Проверяем, что награды отличаются
            if epoch > 2 and check_convergence(rewards, eps):
                patience_counter += 1
                if patience_counter >= patience:
                    flag = True
                    break
            else:
                patience_counter = 0

            #4.  выбираем родителей генотипически
            parents = choose_parents(start_population, cur_rewards, size=parents_number)

            #5. Скрещиваем родительские гены
            population = combine_chromosomes(population=parents, new_size=new_pop, prob=prob)

            #6. Мутируем родительские гены (гены, передающиеся в функцию
            # всё равно ссылаются на свою модель, поэтому мы ничеего не возвращаем)
            mutate_population(population, 0.1)

            start_population = population
            epoch += 1
        print(f"epoch: {epoch}, reward: {max(cur_rewards):.4f}")
    return best_model.eval().state_dict()