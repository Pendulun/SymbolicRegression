from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple
import random
class TreeGenerator(ABC):
    @abstractmethod
    def generate(self, *args) -> Node:
        pass

class Individual():
    def __init__(self, root_node:Node):
        self._root = root_node

    def evaluate(self, data = None):
        return self._root.evaluate(data)
    
    def random_node_and_parent(self) -> Tuple[Node, Node]:
        while True:
            node_stack:List[Node] = list()
            node_stack.append((self.root, None))
            while len(node_stack) > 0:
                curr_node, node_parent = node_stack.pop()
                if random.random() <= curr_node.selection_prob:
                    return curr_node, node_parent
                else:
                    for child in curr_node.childs:
                        node_stack.append((child, curr_node))

    @property
    def depth(self) -> int:
        return self._root.heigth
    
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
    def __init__(self, value = None, selection_prob:float = 1, parent_rule_name:str=None, depth:int = None):
        self._value = value
        self._childs:List[Node] = list()
        self._depth = depth
        self.selection_prob = selection_prob
        self._parent_rule_name = parent_rule_name
    
    def add_child(self, new_child:Node):
        self._childs.append(new_child)
    
    def evaluate(self, *args):
        raise NotImplementedError(f"evaluate not implemented for {self.__class__.__name__} class!")

    def substitute_child(self, old_node:Node, new_node:Node):
        old_node_index = self._childs.index(old_node)
        self._childs[old_node_index] = new_node
    
    @property
    def depth(self) -> int:
        return self._depth
    
    @depth.setter
    def depth(self, new_depth:int):
        raise AttributeError("depth is not subscriptable")
    
    @property
    def heigth(self)->int:
        childs_heigths = [child.heigth for child in self._childs]
        if len(childs_heigths) > 0:
            heigth =  max(childs_heigths)+1
        else:
            heigth = 0

        return heigth
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new_value):
        self._value = new_value
    
    @property
    def childs(self) -> List[Node]:
        return self._childs
    
    @property
    def parent_rule(self)->str:
        return self._parent_rule_name
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        
        return self.value == other.value and self._childs == other._childs
    
    def __str__(self):
        my_str = str(self._value)
        for child in self._childs:
            my_str += " "+str(child)
        
        return my_str