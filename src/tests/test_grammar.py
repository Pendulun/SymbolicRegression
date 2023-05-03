from unittest import main, TestCase
from grammar import Grammar

class TestGrammar(TestCase):

    def test_can_create_grammar(self):
        non_terminal_rules = {
            'expr': [('term','binop','term'), ('unop', 'term')],
            'term': [('var'), ('const')]
        }
        terminal_rules = {
            'var': [('X1'), ('X2'), ('X3')],
            'const': [(1), (2), (3), (4)]
        }
        grammar = Grammar()
        for rule_name, rule_exp in non_terminal_rules.items():
            grammar.add_non_terminal_rule(rule_name, rule_exp)
        
        for rule_name, rule_exp in terminal_rules.items():
            grammar.add_terminal_rule(rule_name, rule_exp)

        self.assertListEqual(grammar.rule('var'), [('X1'),('X2'),('X3')])

if __name__ == "__main__":
    main()