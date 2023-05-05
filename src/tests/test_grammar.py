from unittest import main, TestCase
from grammar import Grammar, Rule, Expansion, NumericExp, StrExp, FuncExpr, Individual
import math

class TestGrammar(TestCase):

    def get_grammar(self)->Grammar:
        non_terminal_rules = [
            Rule('expr', [Expansion(('term','binop','term')), Expansion(('unop', 'term')),
                          Expansion(('expr','binop','expr')), Expansion(('unop', 'expr'))]),
            Rule('term', [StrExp('var'), StrExp('const')])
        ]

        terminal_rules = [
            Rule('var', [StrExp('X1'), StrExp('X2'), StrExp('X3')]),
            Rule('const', [NumericExp(1), NumericExp(2), NumericExp(3), NumericExp(4)]),
            Rule('binop', [FuncExpr(lambda a,b: a+b, '+'), FuncExpr(lambda a,b: a-b, '-')]),
            Rule('unop', [FuncExpr(lambda a:a**2, 'squared'), FuncExpr(lambda a:math.sqrt(a), 'sqrt')])
        ]
        grammar = Grammar()
        for rule in non_terminal_rules:
            grammar.add_non_terminal_rule(rule)
        
        for rule in terminal_rules:
            grammar.add_terminal_rule(rule)
        
        grammar.starting_rule = 'expr'
        
        return grammar

    def test_can_get_rule(self):
        grammar = self.get_grammar()
        expected_terms = [StrExp('X1'), StrExp('X2'), StrExp('X3')]
        self.assertEqual(grammar.rule('var').expansions, expected_terms)
    
    def test_can_print_grammar(self):
        grammar = self.get_grammar()
        
        expected_grammar_str = "expr: term binop term | unop term | expr binop expr | unop expr"
        expected_grammar_str +="\nterm: var | const"
        expected_grammar_str +="\nvar: X1 | X2 | X3"
        expected_grammar_str +="\nconst: 1 | 2 | 3 | 4"
        expected_grammar_str +="\nbinop: + | -"
        expected_grammar_str +="\nunop: squared | sqrt"
        self.assertEqual(str(grammar), expected_grammar_str)
    
    def test_can_set_starting_rule(self):
        grammar = self.get_grammar()
        grammar.starting_rule = 'expr'
        expected_rule = Rule('expr', [Expansion(('term','binop','term')), Expansion(('unop', 'term')),
                                      Expansion(('expr','binop','expr')), Expansion(('unop', 'expr'))])
        self.assertEqual(grammar.starting_rule, expected_rule)
    
    def test_can_expand_given_expansion_idxs_with_binop(self):
        expansion_idxs = [0, 0, 1, 1, 1, 2]
        expected_expansion_str = "(X2 - 3)"
        grammar = self.get_grammar()
        expansion_result = grammar.expand_idxs(expansion_idxs)
        self.assertEqual(expansion_result, expected_expansion_str)
    
    def test_can_expand_given_expansion_idxs_with_unop(self):
        expansion_idxs = [1, 1, 1, 3]
        expected_expansion_str = "(sqrt 4)"
        grammar = self.get_grammar()
        expansion_result = grammar.expand_idxs(expansion_idxs)
        self.assertEqual(expansion_result, expected_expansion_str)
    
    def test_can_expand_given_expansion_idxs_with_unop(self):
        expansion_idxs = [1, 1, 1, 3]
        expected_expansion_str = "(sqrt 4)"
        grammar = self.get_grammar()
        expansion_result = grammar.expand_idxs(expansion_idxs)
        self.assertEqual(expansion_result, expected_expansion_str)
    
    def test_can_expand_given_long_expansion_idxs(self):
        expansion_idxs = [2, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 2]
        expected_expansion_str = "((squared (sqrt X2)) + (X1 - X3))"
        grammar = self.get_grammar()
        expansion_result = grammar.expand_idxs(expansion_idxs)
        self.assertEqual(expansion_result, expected_expansion_str)
    
    def test_can_generate_individual_from_grammar_based_on_expansion_list(self):
        grammar = self.get_grammar()
        expansion_idxs = [2, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 2]
        individual = grammar.individual_from_expansions(expansion_idxs)
        individual_str = str(individual)
        expected_individual_str = "(squared (sqrt (X2)) + (X1 - X3))"
        self.assertEqual(individual_str, expected_individual_str)


if __name__ == "__main__":
    main()