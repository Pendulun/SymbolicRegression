from unittest import main, TestCase
from grammar import Grammar, Rule, Expansion, NumericExp, StrExp, FuncExpr
from grammar import ConstNode, VarNode, UnOPNode, BinOPNode, Individual
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

class TestIndividual(TestCase):
    def test_can_evaluate_const_node(self):
        const_node = ConstNode(3)
        self.assertEqual(const_node.evaluate(), 3)
    
    def test_can_evaluate_var_node(self):
        var_dict = {'X1':2, "X0":4, "Y":7}
        var_node = VarNode("X1")
        self.assertEqual(var_node.evaluate(var_dict), 2)
    
    def test_can_evaluate_unop_node(self):
        parent_node = UnOPNode('squared', lambda a: a**2)
        child_node = ConstNode(3.5)
        parent_node.add_child(child_node)

        self.assertEqual(parent_node.evaluate(), 3.5**2)
    
    def test_can_evaluate_binop_node(self):
        parent_node = BinOPNode('sum', lambda a, b: a+b)
        left_child_node = ConstNode(3.5)
        parent_node.add_child(left_child_node)
        right_child_node = UnOPNode("abs", lambda a: abs(a))
        negative_node = ConstNode(-4)
        right_child_node.add_child(negative_node)
        parent_node.add_child(right_child_node)

        self.assertEqual(parent_node.evaluate(), 3.5 + abs(-4))
    
    def test_can_evaluate_complex_individual_with_data(self):
        data={"X0":1, "X2":-2, "X3":3, "Y":6}
        #((X0-X3) + sqrt(abs(X2)))
        root_node = BinOPNode('+', lambda a,b: a+b)
        l1_left_node = BinOPNode('-', lambda a,b: a-b)
        l1_right_node = UnOPNode('sqrt', lambda a: a**0.5)
        root_node.add_child(l1_left_node)
        root_node.add_child(l1_right_node)

        l2_1_left_node = VarNode("X0")
        l2_1_right_node = VarNode("X3")
        l1_left_node.add_child(l2_1_left_node)
        l1_left_node.add_child(l2_1_right_node)

        l2_2_child_node = UnOPNode('abs', lambda a: abs(a))
        l1_right_node.add_child(l2_2_child_node)

        l3_child_node = VarNode("X2")
        l2_2_child_node.add_child(l3_child_node)

        individual = Individual(root_node)
        expected_result = ((data['X0']-data['X3']) + math.sqrt(abs(data['X2'])))
        self.assertEqual(individual.evaluate(data), expected_result)

if __name__ == "__main__":
    main()