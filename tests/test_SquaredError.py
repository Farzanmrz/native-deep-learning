# Imports
import unittest
import numpy as np
from layers.SquaredError import SquaredError

class TestSquaredError(unittest.TestCase):
    """
    A unit test class for the SquaredError.
    """

    def setUp(self):
        """
        Set up test fixtures, including initializing the SquaredError class and creating test data.
        """
        self.squared_error = SquaredError()
        self.Y = np.array([1, 2, 3, 4])
        self.Yhat = np.array([1.5, 2.5, 3.5, 4.5])

    def test_eval(self):
        """
        Test the eval method to ensure it calculates the squared error loss correctly.
        """
        calculated_loss = self.squared_error.eval(self.Y, self.Yhat)
        expected_loss = np.mean((self.Y - self.Yhat) ** 2)
        self.assertAlmostEqual(calculated_loss, expected_loss, places=7, msg="The calculated loss does not match the expected loss.")

    def test_gradient(self):
        """
        Test the gradient method to ensure it calculates the gradient of the squared error loss correctly.
        """
        calculated_gradient = self.squared_error.gradient(self.Y, self.Yhat)
        expected_gradient = -2 * (self.Y - self.Yhat)
        np.testing.assert_array_almost_equal(calculated_gradient, expected_gradient, err_msg="The calculated gradient does not match the expected gradient.")

if __name__ == '__main__':
    unittest.main()
