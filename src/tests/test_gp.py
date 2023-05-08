from unittest import TestCase, main
from grammar import Grammar, Rule, Expansion, NumericExp, BinFuncExpr, UnFuncExpr, StrExp, VarExpr, Term, FuncTerm
from grammar import GrowIndGenerator
from genetic_prog import GrammarGP, RoulleteSelection
import math

class TestGrammarGP(TestCase):

    def get_grammar(self) -> Grammar:
        non_terminal_rules = [
            Rule('expr', [Expansion((Term('term'),Term('binop'),Term('term'))), 
                          Expansion((Term('unop'), Term('term'))),
                          Expansion((Term('expr'),Term('binop'),Term('expr'))),
                          Expansion((Term('unop'), Term('expr')))]),
            Rule('term', [StrExp(Term('var')), StrExp(Term('const'))]),
            Rule('binop', [BinFuncExpr(FuncTerm("+",lambda a,b: a+b)),
                           BinFuncExpr(FuncTerm('-', lambda a,b: a-b))]),
            Rule('unop', [UnFuncExpr(FuncTerm('squared', lambda a:a**2))])
        ]

        terminal_rules = [
            Rule('var', [VarExpr(Term('X1')), VarExpr(Term('X2')), VarExpr(Term('X3'))]),
            Rule('const', [NumericExp(Term(1)), NumericExp(Term(2)),
                           NumericExp(Term(3)), NumericExp(Term(4))])
        ]
        grammar = Grammar()
        for rule in non_terminal_rules:
            grammar.add_non_terminal_rule(rule)
        
        for rule in terminal_rules:
            grammar.add_terminal_rule(rule)
        
        grammar.starting_rule = 'expr'
        return grammar
    
    def test_generate_n_individuals(self):
        grammar = self.get_grammar()
        ind_generator = GrowIndGenerator(grammar)
        n_individuals = 10
        max_depth = 4
        grammar_gp = GrammarGP(ind_generator, grammar)
        grammar_gp.generate_pop(n_individuals, max_depth)
        self.assertEqual(grammar_gp.n_ind, n_individuals)
    
    def test_roullete_selection(self):
        roullete_selection = RoulleteSelection()
        grammar = self.get_grammar()
        ind_generator = GrowIndGenerator(grammar)
        n_individuals = 10
        max_depth = 4
        grammar_gp = GrammarGP(ind_generator, grammar)
        grammar_gp.generate_pop(n_individuals, max_depth)
        data=[{'X1':3, 'X2':4, 'X3':2, 'Y':9}]
        selected_ind = roullete_selection.select(individuals=grammar_gp.individuals,
                                                 data=data, 
                                                 target='Y',
                                                 fitness_func=lambda a, b: math.sqrt((a-b)**2), 
                                                 k=1, n=1)[0]
        self.assertIn(selected_ind, grammar_gp.individuals)


if __name__ == "__main__":
    main()