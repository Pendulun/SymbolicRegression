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
    
    def compute_depth(self):
        self._root.update_depth(0)
    
    def find_node_and_parent_of_type(self, node_type, max_height:int, other_height:int, other_depth:int) -> Tuple[Node, Node]:
        """
        Returns (None, None) if nothing was found
        """
        node_stack:List[Tuple[Node, Node]] = list()
        node_stack.append((self.root, None))
        nodes_with_parent_of_type_found = list()
        while len(node_stack) > 0:
            curr_node, node_parent = node_stack.pop()
            
            can_substitute_me = curr_node.depth + other_height <= max_height
            can_substitute_other = other_depth + curr_node.height <= max_height
            if isinstance(curr_node, node_type) and can_substitute_me and can_substitute_other:
                nodes_with_parent_of_type_found.append((curr_node, node_parent))
            
            for child in curr_node.childs:
                    node_stack.append((child, curr_node))
        
        if len(nodes_with_parent_of_type_found) > 0:
            return random.choice(nodes_with_parent_of_type_found)
        else:
            return (None, None)

    @property
    def height(self) -> int:
        return self._root.height
    
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

    def update_depth(self, new_depth:int):
        self._depth = new_depth
        for child in self._childs:
            child.update_depth(self._depth+1)
    
    @property
    def depth(self) -> int:
        return self._depth
    
    @depth.setter
    def depth(self, new_depth:int):
        raise AttributeError("depth is not subscriptable")
    
    @property
    def height(self)->int:
        childs_heights = [child.height for child in self._childs]
        if len(childs_heights) > 0:
            height =  max(childs_heights)+1
        else:
            height = 0

        return height
    
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