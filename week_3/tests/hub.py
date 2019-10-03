import unittest

from week_3.tests.mlp_test import TestMultilayerPerceptron


def mlp_test():
    """
    Tests for Programming Assignment 3.4
    :return: number of error and failures
    """
    return run_test(TestMultilayerPerceptron)


def run_test(test_cls):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_cls)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return len(result.errors) + len(result.failures) == 0
