from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
class IndividualGenerator(ABC):
    @abstractmethod
    def generate(self, *args) -> Individual:
        pass

class Individual():
    def __init__(self, root_node:Node):
        self._root = root_node
        self._depth = None

    def evaluate(self, data = None):
        return self._root.evaluate(data)

    @property
    def depth(self) -> int:
        return self._root.depth
    
    @property
    def root(self) -> Node:
        return self._root
    
    def __str__(self):
        return str(self._root)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        
        return self._root == other.root

class Node():
    def __init__(self, value = None):
        self._value = value
        self._childs:List[Node] = list()
        self._depth = None
    
    def add_child(self, new_child:Node):
        self._childs.append(new_child)
    
    def evaluate(self, *args):
        raise NotImplementedError(f"evaluate not implemented for {self.__class__.__name__} class!")

    @property
    def depth(self) -> int:
        if self._depth == None:
            childs_depths = [child.depth for child in self._childs]
            if len(childs_depths) > 0:
                self._depth =  max(childs_depths)+1
            else:
                self._depth = 1

        return self._depth
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new_value):
        self._value = new_value
    
    def __str__(self):
        my_str = str(self._value)
        for child in self._childs:
            my_str += " "+str(child)
        
        return my_str