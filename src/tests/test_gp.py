from unittest import TestCase, main
from grammar import Grammar, Rule, Expansion, NumericExp, BinFuncExp, UnFuncExp, StrExp, VarExpr, Term, FuncTerm
from grammar import GrowTreeGenerator
from genetic_prog import GrammarGP, SelectionFromData, RoulleteSelection, TournamentSelection, LexicaseSelection
from genetic_prog import CrossoverOP, MutationOP
import math
import numpy as np

class TestGrammarGP(TestCase):

    def get_grammar(self) -> Grammar:
        non_terminal_rules = [
            Rule('expr', [Expansion((Term('term'),Term('binop'),Term('term'))), 
                          Expansion((Term('unop'), Term('term'))),
                          Expansion((Term('expr'),Term('binop'),Term('expr'))),
                          Expansion((Term('unop'), Term('expr')))]),
            Rule('term', [StrExp(Term('var')), StrExp(Term('const'))]),
            Rule('binop', [BinFuncExp(FuncTerm("+",lambda a,b: a+b)),
                           BinFuncExp(FuncTerm('-', lambda a,b: a-b))]),
            Rule('unop', [UnFuncExp(FuncTerm('squared', lambda a:a**2))])
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
        ind_generator = GrowTreeGenerator(grammar)
        n_individuals = 10
        max_depth = 4
        grammar_gp = GrammarGP(ind_generator, grammar)
        grammar_gp.generate_pop(n_individuals, max_depth)
        self.assertEqual(grammar_gp.n_ind, n_individuals)
    
    def test_roullete_selection(self):
        roullete_selection = RoulleteSelection()
        grammar = self.get_grammar()
        ind_generator = GrowTreeGenerator(grammar)
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
        self.assertTrue(any([str(ind) == str(selected_ind) for ind in grammar_gp.individuals]))
        selected_ind = roullete_selection.select(individuals=grammar_gp.individuals,
                                                 data=data, 
                                                 target='Y',
                                                 fitness_func=lambda a, b: math.sqrt((a-b)**2), 
                                                 k=1, n=1, better_fitness='lower')[0]
        self.assertTrue(any([str(ind) == str(selected_ind) for ind in grammar_gp.individuals]))
    
    def test_invert_fitness_highest_to_lowest(self):
        fitness_values = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        expected_values = [1, 0.933, 0.8666, 0.80, 0.733, 0.6666]
        new_fitnesses = SelectionFromData.transform_highest_to_lowest(fitness_values)
        for value1, value2 in zip(list(new_fitnesses), expected_values):
            self.assertAlmostEqual(value1, value2, 3)
    
    def test_tournament_selection(self):
        selection_mode = TournamentSelection()
        grammar = self.get_grammar()
        ind_generator = GrowTreeGenerator(grammar)
        n_individuals = 10
        max_depth = 4
        grammar_gp = GrammarGP(ind_generator, grammar)
        grammar_gp.generate_pop(n_individuals, max_depth)
        data=[{'X1':3, 'X2':4, 'X3':2, 'Y':9}]
        selected_ind = selection_mode.select(individuals=grammar_gp.individuals,
                                                 data=data, 
                                                 target='Y',
                                                 fitness_func=lambda a, b: math.sqrt((a-b)**2), 
                                                 k=1, n=1)[0]
        self.assertTrue(any([str(ind) == str(selected_ind) for ind in grammar_gp.individuals]))
        selected_ind = selection_mode.select(individuals=grammar_gp.individuals,
                                                 data=data, 
                                                 target='Y',
                                                 fitness_func=lambda a, b: math.sqrt((a-b)**2), 
                                                 k=2, n=1, better_fitness='lower')[0]
        self.assertTrue(any([str(ind) == str(selected_ind) for ind in grammar_gp.individuals]))
    
    def test_can_calculate_mad(self):
        lexicase_sel = LexicaseSelection()
        ordered_ind_fitnesses = np.array([1, 2, 3, 4, 5, 6, 7])
        median_fit = ordered_ind_fitnesses[len(ordered_ind_fitnesses)//2]
        diff = [val - median_fit for val in ordered_ind_fitnesses]
        abs_diff = [abs(val) for val in diff]
        abs_diff.sort()
        abs_diff_median = abs_diff[len(abs_diff)//2]
        self.assertEqual(lexicase_sel.calculate_mad(ordered_ind_fitnesses), abs_diff_median)
    
    def test_lexicase_sel_is_good_enough(self):
        lexicase_sel = LexicaseSelection()
        ind_fitness = 4.5
        best_fitness = 4
        mad = 1
        self.assertTrue(lexicase_sel.good_enough(ind_fitness, best_fitness, mad))
    
    def test_lexicase_sel_is_not_good_enough(self):
        lexicase_sel = LexicaseSelection()
        ind_fitness = 5
        best_fitness = 4
        mad = 1
        self.assertTrue(lexicase_sel.good_enough(ind_fitness, best_fitness, mad))
        mad=0.5
        self.assertFalse(lexicase_sel.good_enough(ind_fitness, best_fitness, mad))
    
    def test_lexicase_selection(self):
        selection_mode = LexicaseSelection()
        grammar = self.get_grammar()
        ind_generator = GrowTreeGenerator(grammar)
        n_individuals = 10
        max_depth = 4
        grammar_gp = GrammarGP(ind_generator, grammar)
        grammar_gp.generate_pop(n_individuals, max_depth)
        data=[{'X1':3, 'X2':4, 'X3':2, 'Y':9}]
        selected_ind = selection_mode.select(individuals=grammar_gp.individuals,
                                                 data=data, 
                                                 target='Y',
                                                 fitness_func=None,
                                                 n=1)[0]
        self.assertTrue(any([str(ind) == str(selected_ind) for ind in grammar_gp.individuals]))
        selected_ind = selection_mode.select(individuals=grammar_gp.individuals,
                                                 data=data, 
                                                 target='Y',
                                                 fitness_func=None,
                                                 n=1)[0]
        self.assertTrue(any([str(ind) == str(selected_ind) for ind in grammar_gp.individuals]))
    
    def test_mutation_operator(self):
        grammar = self.get_grammar()
        ind_generator = GrowTreeGenerator(grammar)
        grammar_gp = GrammarGP(ind_generator, grammar)
        n_individuals = 1
        max_depth = 4
        grammar_gp.generate_pop(n_individuals, max_depth)
        original_ind = grammar_gp.individuals[0]
        new_ind = MutationOP.mutate(original_ind, grammar, max_depth, ind_generator)
        self.assertTrue(new_ind.depth <= max_depth)

    def test_crossover_operator(self):
        grammar = self.get_grammar()
        ind_generator = GrowTreeGenerator(grammar)
        grammar_gp = GrammarGP(ind_generator, grammar)
        n_individuals = 2
        max_depth = 4
        grammar_gp.generate_pop(n_individuals, max_depth)
        ind1 = grammar_gp.individuals[0]
        ind2 = grammar_gp.individuals[1]
        new_ind1, new_ind2 = CrossoverOP.cross(ind1, ind2, max_depth)
        if new_ind1 is not None:
            self.assertTrue(new_ind1.depth <= max_depth)

        if new_ind2 is not None:
            self.assertTrue(new_ind2.depth <= max_depth)
        
if __name__ == "__main__":
    main()