from grammar import Grammar
from ind_generator import IndividualGenerator
from abc import ABC, abstractmethod

class GP(ABC):
    def __init__(self, ind_generator:IndividualGenerator) -> None:
        self._ind_generator = ind_generator
        self._individuals = list()
    
    @abstractmethod
    def generate_pop(self, n_individuals:int):
        pass

    @property
    def n_ind(self) -> int:
        return len(self._individuals)

class GrammarGP(GP):
    def __init__(self, ind_generator:IndividualGenerator, grammar:Grammar) -> None:
        super().__init__(ind_generator)
        self._grammar = grammar
    
    def generate_pop(self, n_individuals:int, max_depth:int):
        for _ in range(n_individuals):
            self._individuals.append(self._ind_generator.generate(max_depth))

