from grammar import Grammar, Individual
from abc import ABC, abstractmethod
from ind_generator import IndividualGenerator
import numpy as np
import random
from typing import Callable


class GP(ABC):
    def __init__(self, ind_generator:IndividualGenerator) -> None:
        self._ind_generator = ind_generator
        self._individuals:list[Individual] = list()
    
    @abstractmethod
    def generate_pop(self, n_individuals:int):
        pass

    @property
    def n_ind(self) -> int:
        return len(self._individuals)
    
    @property
    def individuals(self) -> list:
        return self._individuals

class GrammarGP(GP):
    def __init__(self, ind_generator:IndividualGenerator, grammar:Grammar) -> None:
        super().__init__(ind_generator)
        self._grammar = grammar
    
    def generate_pop(self, n_individuals:int, max_depth:int):
        for _ in range(n_individuals):
            self._individuals.append(self._ind_generator.generate(max_depth))

class SelectionFromData(ABC):
    @abstractmethod
    def select(self, individuals:list[Individual], data:list[dict], 
               target:str|int|float, fitness_func:Callable, k:int=1, n:int=1) -> list[Individual]:
        pass

class RoulleteSelection(SelectionFromData):
    def select(self, individuals:list[Individual], data:list[dict], 
               target:str|int|float, fitness_func: Callable, k:int=1, n:int=1) -> list[Individual]:
        ind_fitness = np.empty(len(individuals))
        for data_instance in data:
            curr_fitness = np.asarray([fitness_func(ind.evaluate(data_instance), data_instance[target])
                             for ind in individuals])
            ind_fitness = ind_fitness+curr_fitness
        
        selected = [random.choices(population=individuals, weights=ind_fitness, k=1)[0] for _ in range(n)]

        return selected