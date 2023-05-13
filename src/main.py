from genetic_prog import GrammarGP, LexicaseSelection, RoulleteSelection, TournamentSelection
from grammar import Grammar, GrowTreeGenerator
from grammar import Rule, Expansion, Term, StrExp, BinFuncExp, FuncTerm, UnFuncExp, VarExp, NumericExp

import argparse
import math
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm
from typing import List

def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='SymbolicReg')

    parser.add_argument('--train_data_path',
                        required=True,
                        help="The train file path")
    
    parser.add_argument('--test_data_path',
                        required=True,
                        help="The test file path")
    
    parser.add_argument("--target_col",
                        type=str,
                        required=False,
                        default='Y',
                        help="The default data instance col. Default='Y'")
    
    parser.add_argument('--num_inds',
                        type=int,
                        required=True,
                        help="The population size")
    
    parser.add_argument("--num_gens",
                        required=True,
                        type=int,
                        help="The number of generations to run")
    
    parser.add_argument("--selection_type",
                        choices=['Tournament', 'Roullete', 'Lexicase'],
                        required=True,
                        help="The selection algorithm. One of: Tournament, Roullete, Lexicase")
    
    parser.add_argument("--max_height",
                        type=int,
                        required=False,
                        default=7,
                        help="The individuals max height. Default: 7")
    
    parser.add_argument("--selection_k",
                        type=int,
                        required=True,
                        help="The selection sample size")
    
    parser.add_argument("--better_fitness",
                        type=str,
                        choices=['lower', 'greater'],
                        required=False,
                        default='lower',
                        help="The indicative of what is a better fitness. Choices: ['lower', 'greater']\
                            Default: 'lower'")
    
    parser.add_argument("--elitism",
                        action='store_true',
                        required=False,
                        help="A flag that indicates that it should use elitism")
    
    parser.add_argument("--n_cross",
                        type=int,
                        required=True,
                        help="A number indicating how many crossovers to do. Default=0")
    
    
    parser.add_argument("--n_mut",
                        type=int,
                        required=True,
                        help="A number indicating how many mutations to do. Default=0")
    
    parser.add_argument("--p_cross",
                        type=float,
                        default=0.6,
                        help="The crossover probability. Default=0.6")
    
    parser.add_argument("--p_mut",
                        type=float,
                        default=0.3,
                        help="The mutation probability. Default=0.3")
    parser.add_argument("--random_seed",
                        type=float,
                        required=False,
                        default=random.randint(1000),
                        help="The random seed to use. There is no default.")
    return parser
    

def sum_op(a, b) -> float:
    return a + b

def sub_op(a, b) -> float:
    return a - b

def safe_div_op(a, b) -> float:
    if math.isclose(b, 0):
        return a
    
    return a/b

def mul_op(a, b) -> float:
    return a * b

def safe_log_10(a) -> float:
    if a<=0:
        return 0
    
    return math.log10(a)

def configure_grammar(n_features:int) -> Grammar:
    non_terminal_rules = [
            Rule('expr', [Expansion((Term('term'),Term('binop'),Term('term'))), 
                          Expansion((Term('unop'), Term('term'))),
                          Expansion((Term('expr'),Term('binop'),Term('expr'))),
                          Expansion((Term('unop'), Term('expr')))]),
            Rule('term', [StrExp(Term('var')), StrExp(Term('const'))]),
            Rule('binop', [BinFuncExp(FuncTerm("+", sum_op)),
                           BinFuncExp(FuncTerm('-', sub_op)),
                           BinFuncExp(FuncTerm('/*', safe_div_op)),
                           BinFuncExp(FuncTerm('*', mul_op))]),
            Rule('unop', [UnFuncExp(FuncTerm('abs', lambda a:abs(a))),
                          UnFuncExp(FuncTerm('log10', safe_log_10)),
                          UnFuncExp(FuncTerm('cos', math.cos)),
                          UnFuncExp(FuncTerm('sin', math.sin))])
        ]

    terminal_rules = [
        Rule('var', [VarExp(Term(f'X{i}')) for i in range(1, n_features+1)]),
        Rule('const', [NumericExp(Term(i)) for i in range(11)])
    ]

    grammar = Grammar()
    for rule in non_terminal_rules:
        grammar.add_non_terminal_rule(rule)
    
    for rule in terminal_rules:
        grammar.add_terminal_rule(rule)
    
    grammar.starting_rule = 'expr'
    return grammar

def whole_dataset_fitness(a:np.array, b:np.array) -> float:
    simple_errors = a-b
    squared_errors = np.square(simple_errors)
    summed_squared_errors = np.sum(squared_errors)
    mean_squared_errors = summed_squared_errors / len(a)
    root_mean_squared_errors = np.sqrt(mean_squared_errors)
    return root_mean_squared_errors

def evaluate_individual(data, target, fitness_func, individual):
        target_values = np.array([data_instance[target] for data_instance in data])
        curr_ind_predictions = np.array(
            [individual.evaluate(data_instance) for data_instance in data]
            )
        
        return fitness_func(target_values, curr_ind_predictions)

def complete_columns_names(df:pd.DataFrame):
    columns_names = [f'X{i}' for i in range(1, len(df.columns))]
    columns_names.append("Y")
    print(columns_names)
    df.columns = columns_names
    return df

if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()

    random.seed(args.random_seed)
    print(f"Random seed used: {args.random_seed}")

    train_data_df = pd.read_csv(args.train_data_path)
    test_data_df = pd.read_csv(args.test_data_path)

    train_data_df = complete_columns_names(train_data_df)
    test_data_df = complete_columns_names(test_data_df)

    train_data = train_data_df.to_dict(orient='records')
    test_data = test_data_df.to_dict(orient='records')

    single_data_instance_fitness_func = lambda a, b: abs(a-b) 
    if args.selection_type == 'Tournament':
        selector = TournamentSelection
    elif args.selection_type == 'Roullete':
        selector = RoulleteSelection
    elif args.selection_type == 'Lexicase':
        selector = LexicaseSelection

    selection_mode_args = {'k':args.selection_k,
                        'better_fitness':args.better_fitness,
                        'fitness_func':single_data_instance_fitness_func}
    
    whole_dataset_fitness_func = whole_dataset_fitness
    num_features = len(train_data_df.columns)-1
    best_ind_train_fitnesses = list()
    best_ind_test_fitnesses = list()
    times_list = list()
    for run_id in tqdm(range(30)):
        print(f"--------RUN ID: {run_id}")
        grammar = configure_grammar(num_features)
        ind_generator = GrowTreeGenerator(grammar)
        grammar_gp = GrammarGP(ind_generator, grammar)
        grammar_gp.generate_starting_pop(args.num_inds, args.max_height)

        start = time.time()
        best_ind, train_fitness = grammar_gp.adjust(n_generations = args.num_gens, 
                                data = train_data, 
                                target = args.target_col, 
                                selector = selector,
                                selector_args = selection_mode_args, 
                                n_mutations = args.n_mut, 
                                n_crossovers = args.n_cross, 
                                dataset_fitness_func = whole_dataset_fitness_func,
                                elitism = args.elitism, 
                                p_mutation = args.p_mut,
                                p_crossover = args.p_cross, 
                                max_height = args.max_height)
        print(f"Fitness on test: {train_fitness}")
        end = time.time()
        best_ind_train_fitnesses.append(train_fitness)
        total_time = end-start
        print(f"Elapsed time: {total_time} seconds")
        times_list.append(total_time)
        test_fitness = evaluate_individual(data = test_data,
                                                  target = args.target_col,
                                                  fitness_func = whole_dataset_fitness_func,
                                                  individual = best_ind)
        best_ind_test_fitnesses.append(test_fitness)
        print(f"Fitness on test: {test_fitness}")
    
    print(f"Mean time: {sum(times_list)/len(times_list)}")
    