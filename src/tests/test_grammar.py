from unittest import main, TestCase
from grammar import Grammar, Rule, Expansion, NumericExp, BinFuncExp, UnFuncExp, StrExp, VarExpr, Term, FuncTerm
from grammar import ExpansionListTreeGenerator, GrowTreeGenerator
from grammar import ConstNode, VarNode, UnOPNode, BinOPNode, Individual
import math

class TestGrammar(TestCase):

    def setUp(self):
        non_terminal_rules = [
            Rule('expr', [Expansion((Term('term'),Term('binop'),Term('term'))), 
                          Expansion((Term('unop'), Term('term'))),
                          Expansion((Term('expr'),Term('binop'),Term('expr'))),
                          Expansion((Term('unop'), Term('expr')))]),
            Rule('term', [StrExp(Term('var')), StrExp(Term('const'))]),
            Rule('binop', [BinFuncExp(FuncTerm("+",lambda a,b: a+b)),
                           BinFuncExp(FuncTerm('-', lambda a,b: a-b))]),
            Rule('unop', [UnFuncExp(FuncTerm('squared', lambda a:a**2)),
                          UnFuncExp(FuncTerm('sqrt', lambda a:math.sqrt(a)))])
        ]

        terminal_rules = [
            Rule('var', [VarExpr(Term('X1')), VarExpr(Term('X2')), VarExpr(Term('X3'))]),
            Rule('const', [NumericExp(Term(1)), NumericExp(Term(2)),
                           NumericExp(Term(3)), NumericExp(Term(4))])
        ]
        self.grammar = Grammar()
        for rule in non_terminal_rules:
            self.grammar.add_non_terminal_rule(rule)
        
        for rule in terminal_rules:
            self.grammar.add_terminal_rule(rule)
        
        self.grammar.starting_rule = 'expr'

    def test_can_get_rule(self):
        expected_terms = [StrExp(Term('X1')), StrExp(Term('X2')), StrExp(Term('X3'))]
        self.assertEqual(self.grammar.rule('var').expansions, expected_terms)
    
    def test_can_print_grammar(self):
        expected_grammar_str = "expr: term binop term | unop term | expr binop expr | unop expr"
        expected_grammar_str +="\nterm: var | const"
        expected_grammar_str +="\nbinop: + | -"
        expected_grammar_str +="\nunop: squared | sqrt"
        expected_grammar_str +="\nvar: X1 | X2 | X3"
        expected_grammar_str +="\nconst: 1 | 2 | 3 | 4"
        self.assertEqual(str(self.grammar), expected_grammar_str)
    
    def test_can_set_starting_rule(self):
        self.grammar.starting_rule = 'expr'
        expected_rule = Rule('expr', [Expansion((Term('term'),Term('binop'),Term('term'))), 
                          Expansion((Term('unop'), Term('term'))),
                          Expansion((Term('expr'),Term('binop'),Term('expr'))),
                          Expansion((Term('unop'), Term('expr')))])
        
        self.assertEqual(self.grammar.starting_rule, expected_rule)
    
    def test_can_expand_given_expansion_idxs_with_binop(self):
        expansion_idxs = [0, 0, 1, 1, 1, 2]
        expected_expansion_str = "(X2 - 3)"
        expansion_result = self.grammar.expand_idxs(expansion_idxs)
        self.assertEqual(expansion_result, expected_expansion_str)
    
    def test_can_expand_given_expansion_idxs_with_unop(self):
        expansion_idxs = [1, 1, 1, 3]
        expected_expansion_str = "(sqrt 4)"
        expansion_result = self.grammar.expand_idxs(expansion_idxs)
        self.assertEqual(expansion_result, expected_expansion_str)
    
    def test_can_expand_given_expansion_idxs_with_unop(self):
        expansion_idxs = [1, 1, 1, 3]
        expected_expansion_str = "(sqrt 4)"
        expansion_result = self.grammar.expand_idxs(expansion_idxs)
        self.assertEqual(expansion_result, expected_expansion_str)
    
    def test_can_expand_given_long_expansion_idxs(self):
        expansion_idxs = [2, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 2]
        expected_expansion_str = "((squared (sqrt X2)) + (X1 - X3))"
        expansion_result = self.grammar.expand_idxs(expansion_idxs)
        self.assertEqual(expansion_result, expected_expansion_str)
    
class TestIndividualGenerator(TestCase):
    def setUp(self):
        non_terminal_rules = [
            Rule('expr', [Expansion((Term('term'),Term('binop'),Term('term'))), 
                          Expansion((Term('unop'), Term('term'))),
                          Expansion((Term('expr'),Term('binop'),Term('expr'))),
                          Expansion((Term('unop'), Term('expr')))]),
            Rule('term', [StrExp(Term('var')), StrExp(Term('const'))]),
            Rule('binop', [BinFuncExp(FuncTerm("+",lambda a,b: a+b)),
                           BinFuncExp(FuncTerm('-', lambda a,b: a-b))]),
            Rule('unop', [UnFuncExp(FuncTerm('squared', lambda a:a**2)),
                          UnFuncExp(FuncTerm('sqrt', lambda a:math.sqrt(a)))])
        ]

        terminal_rules = [
            Rule('var', [VarExpr(Term('X1')), VarExpr(Term('X2')), VarExpr(Term('X3'))]),
            Rule('const', [NumericExp(Term(1)), NumericExp(Term(2)), 
                           NumericExp(Term(3)), NumericExp(Term(4))])
        ]
        self.grammar = Grammar()
        for rule in non_terminal_rules:
            self.grammar.add_non_terminal_rule(rule)
        
        for rule in terminal_rules:
            self.grammar.add_terminal_rule(rule)
        
        self.grammar.starting_rule = 'expr'
    
    def test_can_generate_individual_from_grammar_based_on_expansion_list(self):
        exp_list_ind_generator = ExpansionListTreeGenerator(self.grammar)

        expansion_idxs = [2, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 2]
        individual = Individual(exp_list_ind_generator.generate(expansion_idxs))
        individual_str = str(individual)
        expected_individual_str = "(squared (sqrt (X2)) + (X1 - X3))"
        self.assertEqual(individual_str, expected_individual_str)
    
    def test_can_generate_individual_grow_method(self):
        grow_ind_generator = GrowTreeGenerator(self.grammar)
        max_depth = 4
        individual = Individual(grow_ind_generator.generate(max_depth))
        self.assertLessEqual(individual.height, max_depth)

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
        #3.5 sum abs(-4)
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