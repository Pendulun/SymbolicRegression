from __future__ import annotations
from typing import List, Tuple, Callable
import random

class Grammar():
    def __init__(self):
        self.all_rules = {}
        self.terminal_rules = list()
        self.non_terminal_rules = list()
        self._starting_rule = ""
    
    def rule(self, rule_str:str)->Rule:
        return self.all_rules[rule_str]
    
    def is_rule(self, rule_str:str) -> bool:
        return rule_str in self.all_rules.keys()
    
    def add_terminal_rule(self, rule:Rule):
        if rule.name not in self.terminal_rules:
            self.terminal_rules.append(rule.name)
        
        #this overrides previous definition
        self.all_rules[rule.name] = rule
    
    def add_non_terminal_rule(self, rule:Rule):
        if rule.name not in self.non_terminal_rules:
            self.non_terminal_rules.append(rule.name)
        
        #this overrides previous definition
        self.all_rules[rule.name] = rule
    
    def random_terminal_rule(self) -> Rule:
        return self.rule(random.choice(self.terminal_rules))
    
    def _expand_idxs_helper(self, rule:Rule, expansions_idx:list) -> str:
        
        curr_exp = rule.at(expansions_idx.pop(0))

        terms_was_list = False
        if type(curr_exp.terms) in [list, tuple]:
            terms = list(curr_exp.terms)
            terms_was_list = True
        else:
            terms = [curr_exp.terms]
        exp_str = ""
        
        for term in terms:
            if self.is_rule(term):
                exp_str += " "+self._expand_idxs_helper(self.rule(term), expansions_idx)
            else:
                exp_str += " "+str(term)
        
        exp_str = exp_str.strip()
        if terms_was_list:
            exp_str = "("+exp_str+")"
        
        return exp_str

    def expand_idxs(self, expansions_idxs:list[int]) -> str:
        return self._expand_idxs_helper(self.starting_rule, list(expansions_idxs))
            
    @property
    def starting_rule(self) -> Rule:
        return self.all_rules[self._starting_rule]
    
    @starting_rule.setter
    def starting_rule(self, new_starting_rule:str):
        self._starting_rule = new_starting_rule
    
    def __str__(self):
        grammar_str = ""
        for rule in self.non_terminal_rules:
            grammar_str += str(self.all_rules[rule])+"\n"
        
        for rule in self.terminal_rules:
            grammar_str += str(self.all_rules[rule])+"\n"
        
        grammar_str = grammar_str.strip('\n')
        return grammar_str

class Rule():
    def __init__(self, name:str, expansions:list[Expansion]):
        self._name = name
        self._expansions = expansions
    
    def at(self,expansion_idx:int) -> Expansion:
        return self.expansions[expansion_idx]
    
    @property
    def expansions(self) -> list[Expansion]:
        return self._expansions
    
    @expansions.setter
    def expansions(self, new_expansions):
        raise AttributeError("expansions is not subscriptable")
    
    @property
    def name(self)->str:
        return self._name

    @name.setter
    def name(self, new_name:str):
        raise AttributeError("Name is not subscriptable")
    
    def __eq__(self, __value: object) -> bool:
        
        if not isinstance(__value, Rule):
            return False
        
        return self.expansions == __value.expansions

    def __str__(self):
        rule_str = self._name+": "
        for expansion in self._expansions:
            rule_str += str(expansion) +" | "
        
        rule_str = rule_str.strip()[:-2]
        return rule_str

    def __len__(self):
        return len(self._expansions)

class Expansion():
    def __init__(self, expansion_terms:List| Tuple):
        self._expansion_terms = expansion_terms
    
    @property
    def terms(self):
        return self._expansion_terms
    
    @terms.setter
    def terms(self, new_terms):
        raise AttributeError("terms is not subscriptable")

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Expansion):
            return False
        
        if __value.terms == self.terms:
            return True
        
        return False

    def __str__(self):
        exp_str = ""
        for term in self._expansion_terms:
            exp_str += str(term) + " "
        
        exp_str = exp_str.strip()
        return exp_str

    def __len__(self):
        return len(self._expansion_terms)

class NumericExp(Expansion):

    def __init__(self, expansion_term: int| float):
        super().__init__([expansion_term])
    
    @property
    def terms(self):
        return self._expansion_terms[0]
    
    def __str__(self):
        return str(self._expansion_terms[0])

class StrExp(Expansion):
        
    def __init__(self, expansion_term: str):
        super().__init__([expansion_term])
    
    @property
    def terms(self):
        return self._expansion_terms[0]
    
    def __str__(self):
        return self._expansion_terms[0]

class FuncExpr(Expansion):
    def __init__(self, expansion_term: Callable, expansion_name:str):
        super().__init__([expansion_name])
        self._expansion_func = expansion_term
    
    @property
    def terms(self):
        return self._expansion_terms[0]
    
    @property
    def func(self) -> Callable:
        return self._expansion_func
    
    @func.setter
    def func(self, new_func:Callable):
        raise AttributeError("func is not subscriptable!")
    
    def __str__(self):
        return self._expansion_terms[0]

class Individual():
    def __init__(self, root_node:Node):
        self._root = root_node
        self._depth = None

    def evaluate(self, data = None):
        return self._root.evaluate(data)

    @property
    def depth(self) -> int:
        return self._root.depth
    
    def __str__(self):
        return str(self._root)

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

class OPNode(Node):
    def __init__(self, value, func:Callable):
        super().__init__(value)
        self._func = func
    
    @property
    def func(self) -> Callable:
        return self._func
    
    @func.setter
    def func(self, new_func:Callable):
        raise AttributeError(f"Func is not subscriptable for {self.__class__.__name__}!")

class BinOPNode(OPNode):
    def __init__(self, value, func:Callable):
        super().__init__(value, func)

    def evaluate(self, *args):
        return self.func(self._childs[0].evaluate(*args), self._childs[1].evaluate(*args))

    def __str__(self):
        my_str = "("+str(self._childs[0])+" "
        my_str += str(self.value)
        my_str += " "+str(self._childs[1])+")"
        return my_str

class UnOPNode(OPNode):
    def __init__(self, value, func:Callable):
        super().__init__(value, func)
    
    def evaluate(self, *args):
        return self.func(self._childs[0].evaluate(*args))
    
    def __str__(self):
        my_str = str(self.value)+" ("
        my_str += str(self._childs[0])+")"
        return my_str

class VarNode(Node):
    def __init__(self, value:None):
        super().__init__(value)
    
    def add_child(self, new_child: Node):
        raise NotImplementedError("I'm a VarNode, I don't have children!")
    
    def evaluate(self, *args):
        vars_dict:dict = args[0]
        return vars_dict[self.value]
    
    def __str__(self):
        return self.value

class ConstNode(Node):
    def __init__(self, value:None):
        super().__init__(value)
    
    def add_child(self, new_child: Node):
        raise NotImplementedError("I'm a ConstNode, I don't have children!")

    def _is_float(self, value: float | int):
        return value % 1 > 0

    def evaluate(self, *args):
        try:
            my_value = float(self.value)
            if self._is_float(my_value):
                return my_value
            else:
                return int(my_value)
        except:
            #It is a string
            return self.value
    
    def __str__(self):
        return str(self.value)

class IndividualGenerator():
    def __init__(self, grammar:Grammar):
        self._grammar = grammar
    
    def generate(self, *args):
        raise NotImplementedError("generate is not implemented!")
    
    @property
    def grammar(self) -> Grammar:
        return self._grammar

    @grammar.setter
    def grammar(self, new_grammar:Grammar):
        raise AttributeError("grammar is not subscriptable!")

class ExpansionListIndGenerator(IndividualGenerator):
    def generate(self, expansion_idxs: list[int]) -> Individual:
        starting_rule = self.grammar.starting_rule
        return Individual(self._generator_helper(starting_rule, list(expansion_idxs)))

    def _generator_helper(self, rule:Rule, expansions_idx:list) -> Node:
        curr_exp = rule.at(expansions_idx.pop(0))

        if type(curr_exp.terms) in [list, tuple]:
            terms = list(curr_exp.terms)

            if len(terms) == 2:
                expansion:FuncExpr = self.grammar.rule(terms[0]).at(expansions_idx.pop(0))
                curr_node_str = expansion.terms
                curr_node_func = expansion.func
                curr_node = UnOPNode(curr_node_str, curr_node_func)
                child_node = self._generator_helper(self.grammar.rule(terms[1]), expansions_idx)
                curr_node.add_child(child_node)
                return curr_node
            elif len(terms) == 3:
                left_child = self._generator_helper(self.grammar.rule(terms[0]), expansions_idx)
                expansion:FuncExpr = self.grammar.rule(terms[1]).at(expansions_idx.pop(0))
                curr_node_str = expansion.terms
                func = expansion.func
                curr_node = BinOPNode(curr_node_str, func)
                right_child = self._generator_helper(self.grammar.rule(terms[2]), expansions_idx)
                curr_node.add_child(left_child)
                curr_node.add_child(right_child)
                return curr_node
            else:
                raise ValueError("Terms size is not 2 or 3!")
        
        else:
            #There is only one value inside the current expansion, that is, a rule
            curr_node_value = self.grammar.rule(curr_exp.terms).at(expansions_idx.pop(0)).terms
            if self._is_a_var_node(curr_node_value):
                return VarNode(curr_node_value)
            else:
                return ConstNode(curr_node_value)
    
    def _is_a_var_node(self, node_value:str) -> bool:
        return node_value[0] == "X"

class GrowIndGenerator(IndividualGenerator):
    def generate(self, max_depth:int) -> Individual:
        starting_rule = self.grammar.starting_rule
        return Individual(self._generator_helper(starting_rule, max_depth))
    
    def _generator_helper(self, rule:Rule, max_depth:int) -> Node:
        if max_depth == 1:
            print("Max depth!")
            curr_rule = self.grammar.random_terminal_rule()
            expansion:Expansion = curr_rule.at(self._random_exp_from_rule(curr_rule))
            curr_node_value = expansion.terms
            print(f"Returning: {curr_node_value}")
            if self._is_a_var_node(curr_node_value):
                return VarNode(curr_node_value)
            else:
                return ConstNode(curr_node_value)
        
        random_exp = self._random_exp_from_rule(rule)
        curr_exp = rule.at(random_exp)

        if type(curr_exp.terms) in [list, tuple]:
            terms = list(curr_exp.terms)

            if len(terms) == 2:
                curr_rule = self.grammar.rule(terms[0])
                expansion:FuncExpr = curr_rule.at(self._random_exp_from_rule(curr_rule))
                curr_node_str = expansion.terms
                curr_node_func = expansion.func
                curr_node = UnOPNode(curr_node_str, curr_node_func)
                child_node = self._generator_helper(self.grammar.rule(terms[1]), max_depth-1)
                curr_node.add_child(child_node)
                return curr_node
            elif len(terms) == 3:
                left_child = self._generator_helper(self.grammar.rule(terms[0]), max_depth-1)
                curr_rule = self.grammar.rule(terms[1])
                expansion:FuncExpr = curr_rule.at(self._random_exp_from_rule(curr_rule))
                curr_node_str = expansion.terms
                func = expansion.func
                curr_node = BinOPNode(curr_node_str, func)
                right_child = self._generator_helper(self.grammar.rule(terms[2]), max_depth-1)
                curr_node.add_child(left_child)
                curr_node.add_child(right_child)
                return curr_node
            else:
                raise ValueError("Terms size is not 2 or 3!")
        
        else:
            #There is only one value inside the current expansion, that is, a rule
            curr_rule = self.grammar.rule(curr_exp.terms)
            curr_node_value = curr_rule.at(self._random_exp_from_rule(curr_rule)).terms
            if self._is_a_var_node(curr_node_value):
                return VarNode(curr_node_value)
            else:
                return ConstNode(curr_node_value)
    
    def _random_exp_from_rule(self, rule:Rule) -> Expansion:
        return random.randint(0, len(rule)-1)
    
    def _is_a_var_node(self, node_value) -> bool:
        if isinstance(node_value, str):
            return node_value[0] == "X" 
        
        return False