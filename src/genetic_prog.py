from grammar import Grammar, Individual
from abc import ABC, abstractmethod
from ind_generator import IndividualGenerator, Individual
import numpy as np
import random
from typing import Callable, Any


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
    
    @staticmethod
    @abstractmethod
    def select(individuals:list[Individual], data:list[dict], 
               target:str|int|float, fitness_func:Callable, k:int=1, n:int=1, better_fitness:str='greater') -> list[Individual]:
        pass

    @staticmethod
    def transform_highest_to_lowest(original_fitnesses:np.array) -> np.array:
        max_fit = original_fitnesses.max()
        min_fit = original_fitnesses.min()
        new_fitness = np.zeros(len(original_fitnesses))
        for fit_idx, fitness in enumerate(original_fitnesses):
            distance_to_max_fit = max_fit - fitness
            new_fitness[fit_idx] = min_fit + distance_to_max_fit
        
        return new_fitness

class RoulleteSelection(SelectionFromData):
    @staticmethod
    def select(individuals:list[Individual], data:list[dict], 
               target:str|int|float, fitness_func: Callable, k:int=1, n:int=1, better_fitness:str='greater') -> list[Individual]:
        """
        This returns n individuals.
        individuals: individuals list
        data: A list of dicts. Example: [{'X1':1, 'X2':2, ..., 'Y':5}, {'X1':2, 'X2':4, ..., 'Y':7}]
        target: The target key for every instance in data. Example: 'Y'
        fitness_func: A callable that must receive the individual evaluation on a data instance and the target value
                        and return the fitness value
        k: The sample size for every selection. Ignored for this selection. Always k=1.
        n: The number of selections to do.
        better_fitness: The logic for the better fitness. Must be one of ['greater','lower']. That is,
            if 'greater' then greater fitness equal better fitness and the equivalent for 'lower'.
        """
        ind_fitness = np.empty(len(individuals))
        for data_instance in data:
            curr_fitness = np.asarray([fitness_func(ind.evaluate(data_instance), data_instance[target])
                             for ind in individuals])
            ind_fitness = ind_fitness+curr_fitness
        
        if better_fitness == 'lower':
            ind_fitness = RoulleteSelection.transform_highest_to_lowest(ind_fitness)
        
        selected = [random.choices(population=individuals, weights=ind_fitness, k=1)[0] for _ in range(n)]

        return selected

class TournamentSelection(SelectionFromData):
    def select(self, individuals: list[Individual], data: list[dict], 
               target: str | int | float, fitness_func: Callable[..., Any], 
               k: int = 1, n: int = 1, better_fitness:str='greater') -> list[Individual]:
        """
        This returns n individuals.
        individuals: individuals list
        data: A list of dicts. Example: [{'X1':1, 'X2':2, ..., 'Y':5}, {'X1':2, 'X2':4, ..., 'Y':7}]
        target: The target key for every instance in data. Example: 'Y'
        fitness_func: A callable that must receive the individual evaluation on a data instance and the target value
                        and return the fitness value
        k: The sample size for every selection
        n: The number of selections to do.
        better_fitness: The logic for the better fitness. Must be one of ['greater','lower']. That is,
            if 'greater' then greater fitness equal better fitness and the equivalent for 'lower'
        """

        n_inds = len(individuals)
        ind_fitness = np.empty(n_inds)
        for data_instance in data:
            curr_fitness = np.asarray([fitness_func(ind.evaluate(data_instance), data_instance[target])
                             for ind in individuals])
            ind_fitness = ind_fitness+curr_fitness

        if better_fitness == 'lower':
            ind_fitness = RoulleteSelection.transform_highest_to_lowest(ind_fitness)
        
        selected = list()
        ind_idxs = range(n_inds)
        for _ in range(n):
            selected_idxs = random.choices(population=ind_idxs, weights=ind_fitness, k=k)
            selected_fitnesses = ind_fitness[selected_idxs]
            max_fitness_idx = np.argmax(selected_fitnesses)
            max_fitness_ind_idx = selected_idxs[max_fitness_idx]
            selected.append(individuals[max_fitness_ind_idx])
        
        return selected