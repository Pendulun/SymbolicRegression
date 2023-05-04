from __future__ import annotations

class Grammar():
    def __init__(self):
        self.all_rules = {}
        self.terminal_rules = list()
        self.non_terminal_rules = list()
    
    def rule(self, rule_str:str)->Rule:
        return self.all_rules[rule_str]
    
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
    
    def __str__(self):
        rule_str = self._name+": "
        for expansion in self._expansions:
            rule_str += str(expansion) +" | "
        
        rule_str = rule_str.strip()[:-2]
        return rule_str

class Expansion():
    def __init__(self, expansion_terms:str|int|float|tuple):
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

class NumericExp(Expansion):
    @property
    def terms(self):
        return self._expansion_term
    
    def __init__(self, expansion_term: int| float):
        self._expansion_term = expansion_term
    
    def __str__(self):
        return str(self._expansion_term)

class StrExp(Expansion):
    @property
    def terms(self):
        return self._expansion_term
    
    def __init__(self, expansion_term: str):
        self._expansion_term = expansion_term
    
    def __str__(self):
        return self._expansion_term