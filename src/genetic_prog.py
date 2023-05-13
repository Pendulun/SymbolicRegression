from __future__ import annotations
from grammar import Grammar, Individual, GrowTreeGenerator
from abc import ABC, abstractmethod
import copy
from ind_generator import TreeGenerator, Individual
import statistics
import numpy as np
import random
from typing import Callable, Any, List, Union, Tuple


class GP(ABC):
    def __init__(self, grammar_tree_generator:TreeGenerator) -> None:
        self._grammar_tree_generator = grammar_tree_generator
        self._individuals:list[Individual] = list()
    
    @abstractmethod
    def generate_starting_pop(self, n_individuals:int):
        pass

    @abstractmethod
    def adjust(self, n_generations:int, data:List[dict], target:str, elitism:bool=True, **args) -> Individual:
        pass

    @property
    def n_ind(self) -> int:
        return len(self._individuals)
    
    @property
    def individuals(self) -> list:
        """
        The current pop
        """
        return self._individuals

class GrammarGP(GP):
    def __init__(self, ind_generator:TreeGenerator, grammar:Grammar) -> None:
        super().__init__(ind_generator)
        self._grammar = grammar
        self._best_fitness_by_gen = list()
        self._worst_fitness_by_gen = list()
        self._std_fitness_by_gen = list()
        self._mean_fitness_by_gen = list()
        self._num_unique_inds_by_gen = list()
        self._num_ind_better_than_parents_after_cross_by_gen = list()
        self._num_ind_worst_than_parents_after_cross_by_gen = list()
        self._saved_dataset_evaluations = 0
        self._num_new_dataset_evaluations = 0
    
    def generate_starting_pop(self, n_individuals:int, max_height:int):
        for _ in range(n_individuals):
            new_ind = Individual(self._grammar_tree_generator.generate(max_height))
            self._individuals.append(new_ind)
    
    def adjust(self, n_generations:int, data:List[dict], target:str, 
               selector:SelectionFromData, selector_args:dict, n_mutations:int, 
               n_crossovers:int, dataset_fitness_func:Callable, elitism:bool=True,
               p_mutation:float=0.5, p_crossover:float=0.5, max_depth:int=5) -> Tuple[Individual, float]:
        """
        This does a symbolic regression. Returns the best individual.
        n_generations: (int) The number of generations to run.
        data: The list of dicts representing the data
        target: (str) The target key of every instance
        selection_mode: A SelectionFromData class
        selection_mode_args: (dict) Params for the selection_mode select method other than individuals, data, target, k and fitness_func
        elitism: (bool) Whether or not to have elitism
        dataset_fitness_func: (Callable) The fitness func to be applied when the individual is evaluated on the whole dataset at once
        max_depth: (int) Individuals max depth
        """
        n_individuals = len(self._individuals)
        self._raise_if_invalid_n_mutations_and_crossovers(n_mutations, n_crossovers, elitism, n_individuals)

        k = selector_args['k']
        better_fitness = selector_args['better_fitness']
        single_data_fitness_func = selector_args['fitness_func']
        n_inds_for_selection_and_ops = n_individuals - 1 if elitism else n_individuals

        for _ in range(n_generations):

            selected_inds = selector.select(self._individuals, data, target, single_data_fitness_func, 
                                                  k=k, n=n_inds_for_selection_and_ops, better_fitness=better_fitness)            

            next_gen_pop:List[Individual] = list()

            new_mutated_inds = self.mutate(n_mutations, p_mutation, max_depth, selected_inds)
            next_gen_pop.extend(new_mutated_inds)

            new_crossed_inds = self.cross(n_crossovers, p_crossover, max_depth, selected_inds, 
                                          data, target, dataset_fitness_func, better_fitness)
            next_gen_pop.extend(new_crossed_inds)

            ind_fitnesess = self._evaluate_gen_individuals(data, target, dataset_fitness_func)
            if better_fitness == "lower":
                best_fit_idx = np.argmin(ind_fitnesess)
                worst_fit_idx = np.argmax(ind_fitnesess)
            else:
                best_fit_idx = np.argmax(ind_fitnesess)
                worst_fit_idx = np.argmin(ind_fitnesess)

            self._save_statistics(ind_fitnesess, best_fit_idx, worst_fit_idx)
          
            if elitism:
               next_gen_pop.append(self._individuals[best_fit_idx])
            
            self._individuals = next_gen_pop
        
        ind_fitnesess = self._evaluate_gen_individuals(data, target, dataset_fitness_func)
        if better_fitness == "lower":
            best_fit_idx = np.argmin(ind_fitnesess)
            best_fit = np.min(ind_fitnesess)
        else:
            best_fit_idx = np.argmax(ind_fitnesess)
            best_fit = np.max(ind_fitnesess)

        return self._individuals[best_fit_idx], best_fit

    def _raise_if_invalid_n_mutations_and_crossovers(self, n_mutations, n_crossovers, elitism, n_individuals):
        expected_n_ops = n_individuals if not elitism else n_individuals -1
        if n_mutations + 2*n_crossovers != expected_n_ops:
            raise ValueError(f"Invalid values for n_mutations and n_crossovers! \
            n_mutations + 2*n_crossovers is {n_mutations + 2*n_crossovers} \
            but {expected_n_ops} was expected with elitism equals {elitism}!")

    def _save_statistics(self, ind_fitnesess, best_fit_idx, worst_fit_idx):
        self._best_fitness_by_gen.append(ind_fitnesess[best_fit_idx])
        self._worst_fitness_by_gen.append(ind_fitnesess[worst_fit_idx])
        self._mean_fitness_by_gen.append(np.mean(ind_fitnesess))
        self._std_fitness_by_gen.append(np.std(ind_fitnesess))
        self._num_unique_inds_by_gen.append(len(set(ind_fitnesess)))

    def cross(self, n_crossovers, p_crossover, max_depth, selected_inds, data:List[dict], 
              target:str, dataset_fitness_func:Callable, better_fitness:str) -> List[Individual]:
        new_crossed_inds = list()
        num_childs_better_than_parents = 0
        num_childs_worst_than_parents = 0
        for _ in range(n_crossovers):
            random_ind1, random_ind2 = random.choices(selected_inds, k=2)
            if random.random() < p_crossover:
                new_ind1, new_ind2 = CrossoverOP.cross(random_ind1, random_ind2, max_depth)

                crossover_happened = new_ind1 is not None
                if crossover_happened:
                    new_crossed_inds.append(new_ind1)
                    new_crossed_inds.append(new_ind2)

                    parents_fits = self._evaluate_individuals(data, target, dataset_fitness_func, [random_ind1, random_ind2])
                    parents_mean_fit = np.mean(parents_fits)
                    new_inds_fits = self._evaluate_individuals(data, target, dataset_fitness_func, [new_ind1, new_ind2])

                    for new_ind_fit in new_inds_fits:
                        if better_fitness == 'lower':
                            if new_ind_fit < parents_mean_fit:
                                num_childs_better_than_parents += 1
                            else:
                                num_childs_worst_than_parents +=1
                        else:
                            if new_ind_fit > parents_mean_fit:
                                num_childs_better_than_parents += 1
                            else:
                                num_childs_worst_than_parents +=1

                else:
                    new_crossed_inds.append(copy.deepcopy(random_ind1))
                    new_crossed_inds.append(copy.deepcopy(random_ind2))
            else:
                new_crossed_inds.append(copy.deepcopy(random_ind1))
                new_crossed_inds.append(copy.deepcopy(random_ind2))

        self._num_ind_better_than_parents_after_cross_by_gen.append(num_childs_better_than_parents)
        self._num_ind_worst_than_parents_after_cross_by_gen.append(num_childs_worst_than_parents)
        return new_crossed_inds

    def mutate(self, n_mutations, p_mutation, max_depth, selected_inds) -> List[Individual]:
        new_mutated_inds = list()
        for _ in range(n_mutations):
            random_ind = random.choice(selected_inds)
            if random.random() < p_mutation:
                new_ind = MutationOP.mutate(random_ind, self._grammar, max_depth, self._grammar_tree_generator)
                new_ind._dataset_fitness = None
                new_mutated_inds.append(new_ind)
            else:
                new_mutated_inds.append(copy.deepcopy(random_ind))
        return new_mutated_inds

    def _evaluate_gen_individuals(self, data:List[dict], target:str, fitness_func:Callable) -> np.ndarray:
        individuals = self._individuals
        ind_fitnesses = self._evaluate_individuals(data, target, fitness_func, individuals)
        return ind_fitnesses

    def _evaluate_individuals(self, data, target, fitness_func, individuals:list[Individual]):
        ind_fitnesses = np.empty(len(individuals))
        target_values = np.array([data_instance[target] for data_instance in data])

        for ind_idx, ind in enumerate(individuals):

            if not ind.was_evaluated_in_whole_dataset():
                curr_ind_predictions = np.array(
                    [ind.evaluate(data_instance) for data_instance in data]
                    )
                ind_fitness = fitness_func(target_values, curr_ind_predictions)
                self._num_new_dataset_evaluations += 1
                ind._dataset_fitness = ind_fitness
            else:
                self._saved_dataset_evaluations += 1
                ind_fitness = ind._dataset_fitness
            
            ind_fitnesses[ind_idx] = ind_fitness
        return ind_fitnesses

class SelectionFromData(ABC):
    
    @staticmethod
    @abstractmethod
    def select(individuals:List[Individual], data:List[dict], 
               target:Union[str,int,float], fitness_func:Callable, k:int=1, n:int=1, better_fitness:str='greater') -> List[Individual]:
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
        pass

    @staticmethod
    def transform_highest_to_lowest(original_fitnesses:np.array) -> np.ndarray:
        """
        Does this: original=[0, 0.2, 0.4, 0.6, 0.8, 1], transformed=[1, 0.933, 0.8666, 0.80, 0.733, 0.6666]
        """
        return 1 -(original_fitnesses/np.sum(original_fitnesses))

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
        return ind_fitness <= best_fitness + mad

    def calculate_mad(self, ind_fitnesses:np.array):
        error_median = statistics.median(ind_fitnesses)
        mad = statistics.median(np.abs(ind_fitnesses - error_median))
        return mad


    def _evaluate_individuals(self, individuals:List[Individual], target:str, run_inds_idxs:List, sample:dict) -> np.ndarray:
        ind_fitnesses:np.array = np.zeros(len(run_inds_idxs))
        for considered_ind_idx, real_ind_idx in enumerate(run_inds_idxs):
            individual = individuals[real_ind_idx]
            predicted = individual.evaluate(sample)
            error = abs(sample[target] - predicted)
            ind_fitnesses[considered_ind_idx] = error
        
        return ind_fitnesses

class MutationOP():
    @staticmethod
    def mutate(individual:Individual, grammar:Grammar, max_height:int, tree_generator:GrowTreeGenerator):
        """
        This mutation operator assumes that any node can replace another with itself and its sub-tree
        """
        ind_copy = copy.deepcopy(individual)
        random_node, parent_node = ind_copy.random_node_and_parent()
        curr_depth = random_node.depth
        #Assumes that a node can be replaced by any other node
        starting_rule = grammar.rule(random_node.parent_rule)

        new_node = tree_generator.generate(max_height=max_height,
                                           starting_rule=starting_rule,
                                           curr_depth=curr_depth)

        if parent_node is not None:
            parent_node.substitute_child(random_node, new_node)
            ind_copy.compute_depth()
            return ind_copy
        else:
            #random_node is root_node
            new_ind = Individual(new_node)
            new_ind.compute_depth()
            return new_ind

class CrossoverOP():
    @staticmethod
    def cross(ind1:Individual, ind2:Individual, max_height:int) -> Tuple[Individual, Individual]:
        ind1_copy = copy.deepcopy(ind1)
        ind2_copy = copy.deepcopy(ind2)
        
        node1, parent_node1 = ind1_copy.random_node_and_parent()
        node2, parent_node2 = ind2_copy.find_node_and_parent_of_type(type(node1), max_height, node1.height, node1.depth)
        
        if CrossoverOP.found_a_equivalent_node(node2):
            if CrossoverOP.node_child_isnt_root(parent_node2):
                parent_node2.substitute_child(node2, node1)
            else:
                ind2_copy = Individual(node1)
            
            if CrossoverOP.node_child_isnt_root(parent_node1):
                parent_node1.substitute_child(node1, node2)
            else:
                ind1_copy = Individual(node2)
            
            ind1_copy.compute_depth()
            ind2_copy.compute_depth()

            ind1_copy._dataset_fitness = None
            ind2_copy._dataset_fitness = None

            return ind1_copy, ind2_copy
        else:
            #crossover could not be applied
            return None, None

    @staticmethod
    def node_child_isnt_root(parent_node2):
        return parent_node2 is not None

    @staticmethod
    def found_a_equivalent_node(node2):
        return node2 is not None