from grammar import Grammar, Individual, GrowTreeGenerator
from abc import ABC, abstractmethod
import copy
from ind_generator import TreeGenerator, Individual
import statistics
import numpy as np
import random
from typing import Callable, Any, List, Union


class GP(ABC):
    def __init__(self, grammar_tree_generator:TreeGenerator) -> None:
        self._grammar_tree_generator = grammar_tree_generator
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
    def __init__(self, ind_generator:TreeGenerator, grammar:Grammar) -> None:
        super().__init__(ind_generator)
        self._grammar = grammar
    
    def generate_pop(self, n_individuals:int, max_depth:int):
        for _ in range(n_individuals):
            new_ind = Individual(self._grammar_tree_generator.generate(max_depth))
            self._individuals.append(new_ind)

class SelectionFromData(ABC):
    
    @staticmethod
    @abstractmethod
    def select(individuals:List[Individual], data:List[dict], 
               target:Union[str,int,float], fitness_func:Callable, k:int=1, n:int=1, better_fitness:str='greater') -> List[Individual]:
        pass

    @staticmethod
    def transform_highest_to_lowest(original_fitnesses:np.array) -> np.array:
        """
        Does this: original=[0, 0.2, 0.4, 0.6, 0.8, 1], transformed=[1, 0.933, 0.8666, 0.80, 0.733, 0.6666]
        """
        return 1 -(original_fitnesses/original_fitnesses.sum())

class RoulleteSelection(SelectionFromData):
    @staticmethod
    def select(individuals:List[Individual], data:List[dict], 
               target:Union[str, int, float], fitness_func: Callable, k:int=1, n:int=1, better_fitness:str='greater') -> List[Individual]:
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
        
        selected_inds = random.choices(population=individuals, weights=ind_fitness, k=n)
        selected_inds = [copy.deepcopy(ind) for ind in selected_inds]

        return selected_inds

class TournamentSelection(SelectionFromData):
    def select(self, individuals: List[Individual], data: List[dict], 
               target: Union[str, int, float], fitness_func: Callable[..., Any], 
               k: int = 1, n: int = 1, better_fitness:str='greater') -> List[Individual]:
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
            selected.append(copy.deepcopy(individuals[max_fitness_ind_idx]))
        
        return selected

class LexicaseSelection(SelectionFromData):
    def select(self, individuals: List[Individual], data: List[dict], 
               target: Union[str, int, float], fitness_func: Callable[..., Any], 
               k: int = 1, n: int = 1, better_fitness:str='greater') -> List[Individual]:
        """
        This returns n individuals.
        individuals: individuals list
        data: A list of dicts. Example: [{'X1':1, 'X2':2, ..., 'Y':5}, {'X1':2, 'X2':4, ..., 'Y':7}]
        target: The target key for every instance in data. Example: 'Y'
        fitness_func: A callable that must receive the individual evaluation on a data instance and the target value
                        and return the fitness value. Ignored for this selection type
        k: The sample size for every selection. Ignored for this selection type
        n: The number of selections to do.
        better_fitness: The logic for the better fitness. Must be one of ['greater','lower']. That is,
            if 'greater' then greater fitness equal better fitness and the equivalent for 'lower'. Ignored for this selection type.
        """
        selected_individuals = list()

        for _ in range(n):
            data_samples = random.sample(data, k=len(data))
            good_ind_idxs = range(len(individuals))
            
            for sample in data_samples:
                ind_fitnesses:np.array = self._evaluate_individuals(individuals, target, good_ind_idxs, sample)

                selected_good_ind_idxs = self.select_good_enough_inds_idxs(good_ind_idxs, ind_fitnesses)

                if len(selected_good_ind_idxs) == 0:
                    good_ind_idxs = random.choice(good_ind_idxs)
                    print("WARNING: selected_good_ind_idx was empty! Choosing a random ind!")
                    break
                
                good_ind_idxs = selected_good_ind_idxs

                if len(good_ind_idxs) == 1:
                    #Filtered enough individuals
                    break
            
            selected_ind_idx = random.choice(good_ind_idxs)
            selected_individuals.append(copy.deepcopy(individuals[selected_ind_idx]))

        return selected_individuals

    def select_good_enough_inds_idxs(self, run_inds_idxs:list, ind_fitnesses:np.array) -> list:
        mad = self.calculate_mad(ind_fitnesses)
        next_run_ind_idxs = list()
        best_fitness = ind_fitnesses.min()

        next_run_ind_idxs = [run_inds_idxs[ind_idx]
                                        for ind_idx, ind_fitness in enumerate(ind_fitnesses)
                                        if self.good_enough(ind_fitness, best_fitness, mad)]
        run_inds_idxs = next_run_ind_idxs
        return run_inds_idxs

    def good_enough(self, ind_fitness, best_fitness, mad) -> bool:
        return ind_fitness < best_fitness + mad

    def calculate_mad(self, ind_fitnesses:np.array):
        error_median = statistics.median(ind_fitnesses)
        mad = statistics.median(np.abs(ind_fitnesses - error_median))
        return mad


    def _evaluate_individuals(self, individuals:List[Individual], target:str, run_inds_idxs:List, sample:dict) -> np.array:
        ind_fitnesses:np.array = np.zeros(len(run_inds_idxs))
        for considered_ind_idx, real_ind_idx in enumerate(run_inds_idxs):
            individual = individuals[real_ind_idx]
            predicted = individual.evaluate(sample)
            error = abs(sample[target] - predicted)
            ind_fitnesses[considered_ind_idx] = error
        
        return ind_fitnesses

class MutationOP():
    @staticmethod
    def mutate(individual:Individual, grammar:Grammar, max_depth:int, tree_generator:GrowTreeGenerator):
        """
        This mutation operator assumes that any node can replace another with itself and its sub-tree
        """
        random_node, parent_node = individual.random_node_and_parent()
        curr_depth = random_node.depth

        #Assumes that a node can be replaced by any other node
        starting_rule = grammar.rule(random_node.parent_rule)

        new_node = tree_generator.generate(max_depth=max_depth,
                                           starting_rule=starting_rule,
                                           curr_depth=curr_depth)

        if parent_node is not None:
            parent_node.substitute_child(random_node, new_node)
        else:
            #random_node is root_node
            individual = Individual(new_node)

        return individual