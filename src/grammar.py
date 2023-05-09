from __future__ import annotations
from ind_generator import TreeGenerator, Node, Individual
from typing import List, Tuple, Callable, Any
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
    
    def random_non_terminal_exp_from_rule(self, rule_str:str) -> Rule:
        rule = self.rule(rule_str)
        rule_expansions = rule.expansions
        non_terminal_expansions = [exp for exp in rule_expansions if not exp.has_terminal()]
        return self.rule(random.choice(non_terminal_expansions))
    
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
            if self.is_rule(term.value):
                exp_str += " "+self._expand_idxs_helper(self.rule(term.value), expansions_idx)
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
    
    def at(self, expansion_idx:int) -> Expansion:
        return self.expansions[expansion_idx]
    
    def random_expansion(self) -> Expansion:
        return self.at(random.randint(0, len(self._expansions)-1))
    
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
    def __init__(self, expansion_terms:List[Term]| Tuple[Term]):
        self._expansion_terms = expansion_terms
    
    @property
    def terms(self):
        return self._expansion_terms
    
    @terms.setter
    def terms(self, new_terms):
        raise AttributeError("terms is not subscriptable")

    def to_node(self, parent_rule_name:str) -> Node:
        raise NotImplementedError(f"to_node is not implemented for {self.__class__.__name__}!")

    def has_terminal(self) -> bool:
        return any([term.is_terminal() for term in self.terms])

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

    def __init__(self, term: Term):
        super().__init__([term])
    
    @property
    def terms(self):
        return self._expansion_terms[0]
    
    def to_node(self, parent_rule_name:str) -> Node:
        return ConstNode(self.terms.value, parent_rule_name=parent_rule_name)
    
    def __str__(self):
        return str(self._expansion_terms[0].value)

class StrExp(Expansion):
        
    def __init__(self, term: Term):
        super().__init__([term])
    
    @property
    def terms(self):
        return self._expansion_terms[0]
    
    def __str__(self):
        return self._expansion_terms[0].value

class VarExpr(StrExp):
    def __init__(self, term: Term):
        super().__init__(term)

    def to_node(self, parent_rule_name:str) -> Node:
        return VarNode(self.terms.value, parent_rule_name=parent_rule_name)

class FuncExp(Expansion):
    def __init__(self, term:FuncTerm):
        super().__init__([term])
        self._expansion_func = term.func
    
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
        return self._expansion_terms[0].value

class UnFuncExp(FuncExp):
    def __init__(self, term:FuncTerm):
        super().__init__(term)
    
    def to_node(self, parent_rule_name:str) -> Node:
        return UnOPNode(self.terms, self.func, parent_rule_name=parent_rule_name)

class BinFuncExp(FuncExp):
    def __init__(self, term:FuncTerm):
        super().__init__(term)
    
    def to_node(self, parent_rule_name:str) -> Node:
        return BinOPNode(self.terms, self.func, parent_rule_name=parent_rule_name)

class Term:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, new_value):
        raise AttributeError("value is not subscriptable!")
    
    def is_terminal(self) -> bool:
        raise NotImplementedError(f"is_terminal not implemented for {self.__class__.__name__}!")
    
    def __str__(self):
        return str(self.value)
    
    def __repr__(self) -> str:
        return f'Term({self.value})'
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            return False
        
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(self.value)

class TerminalTerm(Term):

    def __init__(self, value):
        super().__init__(value)
    
    def is_terminal(self) -> bool:
        return True
    
class NonTerminalTerm(Term):
    def __init__(self, value):
        super().__init__(value)
    
    def is_terminal(self) -> bool:
        return False

class FuncTerm(NonTerminalTerm):
    def __init__(self, value, func:Callable):
        super().__init__(value)
        self._func = func
    
    @property
    def func(self) -> Callable:
        return self._func

    @func.setter
    def func(self, new_func:Callable):
        raise AttributeError("func is not subscriptable!")
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FuncTerm):
            return False
        
        return (self.value, self.func) == (other.value, other.func)
    
    def __repr__(self) -> str:
        return f"FuncTerm({self.value}, {self.func})"
    
    def __hash__(self) -> int:
        return hash((self.value, self.func))
    
    def __str__(self):
        return str(self.value)

class OPNode(Node):
    def __init__(self, value, func:Callable, selection_prob:float=1, parent_rule_name:str=None):
        super().__init__(value, selection_prob, parent_rule_name)
        self._func = func
    
    @property
    def func(self) -> Callable:
        return self._func
    
    @func.setter
    def func(self, new_func:Callable):
        raise AttributeError(f"Func is not subscriptable for {self.__class__.__name__}!")

class BinOPNode(OPNode):
    def __init__(self, value, func:Callable, selection_prob:float=1, parent_rule_name:str=None):
        super().__init__(value, func, selection_prob, parent_rule_name)

    def evaluate(self, *args):
        return self.func(self._childs[0].evaluate(*args), self._childs[1].evaluate(*args))

    def __str__(self):
        my_str = "("+str(self._childs[0])+" "
        my_str += str(self.value)
        my_str += " "+str(self._childs[1])+")"
        return my_str

class UnOPNode(OPNode):
    def __init__(self, value, func:Callable, selection_prob:float=1, parent_rule_name:str=None):
        super().__init__(value, func, selection_prob, parent_rule_name)
    
    def evaluate(self, *args):
        return self.func(self._childs[0].evaluate(*args))
    
    def __str__(self):
        my_str = str(self.value)+" ("
        my_str += str(self._childs[0])+")"
        return my_str

class VarNode(Node):
    def __init__(self, value, selection_prob:float=1, parent_rule_name:str=None):
        super().__init__(value, selection_prob, parent_rule_name)
    
    def add_child(self, new_child: Node):
        raise NotImplementedError("I'm a VarNode, I don't have children!")
    
    def evaluate(self, *args):
        vars_dict:dict = args[0]
        return vars_dict[self.value]
    
    def __str__(self):
        return self.value

class ConstNode(Node):
    def __init__(self, value:None, selection_prob:float=1, parent_rule_name:str=None):
        super().__init__(value, selection_prob, parent_rule_name)
    
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

class GrammarTreeGenerator(TreeGenerator):
    def __init__(self, grammar:Grammar):
        self._grammar = grammar
    
    def generate(self, *args) -> Node:
        raise NotImplementedError("generate is not implemented!")
    
    @property
    def grammar(self) -> Grammar:
        return self._grammar

    @grammar.setter
    def grammar(self, new_grammar:Grammar):
        raise AttributeError("grammar is not subscriptable!")

class ExpansionListTreeGenerator(GrammarTreeGenerator):
    def generate(self, expansion_idxs: list[int], starting_rule:Rule=None) -> Node:
        if starting_rule == None:
            starting_rule = self.grammar.starting_rule
        return self._generator_helper(starting_rule, list(expansion_idxs))
    
    def _generator_helper(self, rule:Rule, expansions_idx:list) -> Node:
        curr_exp = rule.at(expansions_idx.pop(0))

        if type(curr_exp.terms) in [list, tuple]:
            terms = list(curr_exp.terms)

            if len(terms) == 2:
                curr_node = self._new_node_from_expansion_of_rule(expansions_idx.pop(0), terms[0].value)
                child_node = self._generator_helper(self.grammar.rule(terms[1].value), expansions_idx)
                curr_node.add_child(child_node)
                return curr_node
            elif len(terms) == 3:
                left_child = self._generator_helper(self.grammar.rule(terms[0].value), expansions_idx)
                curr_node = self._new_node_from_expansion_of_rule(expansions_idx.pop(0), terms[1].value)
                right_child = self._generator_helper(self.grammar.rule(terms[2].value), expansions_idx)
                curr_node.add_child(left_child)
                curr_node.add_child(right_child)
                return curr_node
            else:
                raise ValueError("Terms size is not 2 or 3!")
        
        else:
            #There is only one value inside the current expansion, that is, a rule
            return self._new_node_from_expansion_of_rule(expansions_idx.pop(0), curr_exp.terms.value)

    def _new_node_from_expansion_of_rule(self, expansion_idx, rule_name):
        curr_rule = self.grammar.rule(rule_name)
        expansion = curr_rule.at(expansion_idx)
        curr_node = expansion.to_node(curr_rule.name)
        return curr_node

class GrowTreeGenerator(GrammarTreeGenerator):
    def generate(self, max_depth:int, starting_rule:Rule=None) -> Node:
        if starting_rule == None:
            starting_rule = self.grammar.starting_rule
        return self._generator_helper(starting_rule, max_depth)
    
    def _generator_helper(self, rule:Rule, max_depth:int) -> Node:
        if max_depth == 1:
            curr_rule = self.grammar.random_terminal_rule()
            expansion = curr_rule.random_expansion()
            return expansion.to_node(curr_rule.name)
        
        curr_exp = rule.random_expansion()

        if type(curr_exp.terms) in [list, tuple]:
            terms = list(curr_exp.terms)

            if len(terms) == 2:
                curr_node = self.new_node_from_random_expansion_of_rule(terms[0].value)
                child_node = self._generator_helper(self.grammar.rule(terms[1].value), max_depth-1)
                curr_node.add_child(child_node)
                return curr_node
            elif len(terms) == 3:
                left_child = self._generator_helper(self.grammar.rule(terms[0].value), max_depth-1)
                curr_rule = self.grammar.rule(terms[1].value)
                expansion = curr_rule.random_expansion()
                curr_node = expansion.to_node(curr_rule.name)
                right_child = self._generator_helper(self.grammar.rule(terms[2].value), max_depth-1)
                curr_node.add_child(left_child)
                curr_node.add_child(right_child)
                return curr_node
            else:
                raise ValueError("Terms size is not 2 or 3!")
        
        else:
            #There is only one value inside the current expansion, that is, a rule
            curr_rule = self.grammar.rule(curr_exp.terms.value)
            expansion = curr_rule.random_expansion()
            return expansion.to_node(curr_rule.name)

    def new_node_from_random_expansion_of_rule(self, rule_name:str):
        curr_rule = self.grammar.rule(rule_name)
        expansion = curr_rule.random_expansion()
        curr_node = expansion.to_node(curr_rule.name)
        return curr_node