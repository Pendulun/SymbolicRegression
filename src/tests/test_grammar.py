from unittest import main, TestCase
from grammar import Grammar, Rule, Expansion, NumericExp, StrExp

class TestGrammar(TestCase):

    def get_grammar(self)->Grammar:
        non_terminal_rules = [
            Rule('expr', [Expansion(('term','binop','term')), Expansion(('unop', 'term'))]),
            Rule('term', [StrExp('var'), StrExp('const')])
        ]

        terminal_rules = [
            Rule('var', [StrExp('X1'), StrExp('X2'), StrExp('X3')]),
            Rule('const', [NumericExp(1), NumericExp(2), NumericExp(3), NumericExp(4)])
        ]
        grammar = Grammar()
        for rule in non_terminal_rules:
            grammar.add_non_terminal_rule(rule)
        
        for rule in terminal_rules:
            grammar.add_terminal_rule(rule)
        
        return grammar

    def test_can_get_rule(self):
        grammar = self.get_grammar()
        expected_terms = [StrExp('X1'), StrExp('X2'), StrExp('X3')]
        self.assertEqual(grammar.rule('var').expansions, expected_terms)
    
    def test_can_print_grammar(self):
        grammar = self.get_grammar()
        
        expected_grammar_str = "expr: term binop term | unop term\nterm: var | const\nvar: X1 | X2 | X3\nconst: 1 | 2 | 3 | 4"
        self.assertEqual(str(grammar), expected_grammar_str)


if __name__ == "__main__":
    main()