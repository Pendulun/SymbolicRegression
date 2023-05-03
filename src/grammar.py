class Grammar():
    def __init__(self):
        self.all_rules = {}
        self.terminal_rules = list()
        self.non_terminal_rules = list()
    
    def rule(self, rule_str:str)->list:
        return self.all_rules[rule_str]
    
    def add_terminal_rule(self, rule_name:str, rule_expansions:list):
        if rule_name not in self.terminal_rules:
            self.terminal_rules.append(rule_name)
        
        #this overrides previous definition
        self.all_rules[rule_name] = rule_expansions
    
    def add_non_terminal_rule(self, rule_name:str, rule_expansions:list):
        if rule_name not in self.non_terminal_rules:
            self.non_terminal_rules.append(rule_name)
        
        #this overrides previous definition
        self.all_rules[rule_name] = rule_expansions