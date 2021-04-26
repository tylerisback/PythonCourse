import unittest
import linear_analysis
import numpy as np

class MyTestCase(unittest.TestCase):

    #We can look from different variables to calculate the effect of other variables
    #but dropped column and target variable must be same.
    def test_the_function1(self):
        target, dropped_column, VarOfInterest_name = 'Average Family Size', 'Districts', 'Distances'
        all_B, standard_error_B, t_confidence_intervals, VarOfInterest= linear_analysis.the_function(target, dropped_column, VarOfInterest_name)

        self.assertEqual(target, 'Average Family Size')
        self.assertEqual(dropped_column, 'Districts')
        self.assertEqual(VarOfInterest_name, 'Distances')

    def test_the_function2(self):
        target, dropped_column, VarOfInterest_name = 'Average Family Size', 'Districts', 'Waste'
        all_B, standard_error_B, t_confidence_intervals, VarOfInterest= linear_analysis.the_function(target, dropped_column, VarOfInterest_name)

        self.assertEqual(target, 'Average Family Size')
        self.assertEqual(dropped_column, 'Districts')
        self.assertEqual(VarOfInterest_name, 'Waste')

    def test_the_function3(self):
        target, dropped_column, VarOfInterest_name = 'Average Family Size', 'Districts', 'GasConsumption'
        all_B, standard_error_B, t_confidence_intervals, VarOfInterest= linear_analysis.the_function(target, dropped_column, VarOfInterest_name)
        self.assertEqual(target, 'Average Family Size')
        self.assertEqual(dropped_column, 'Districts')
        self.assertEqual(VarOfInterest_name, 'GasConsumption')

if __name__ == '__main__':
    unittest.main()
