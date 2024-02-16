# Imports
import unittest
import numpy as np
from layers.LogLoss import LogLoss  # Adjust import according to your project structure

class TestLogLoss(unittest.TestCase):
    """
    A unit test class for the LogLoss function.
    """

    def setUp(self):
        """
        Set up test fixtures, including initializing the LogLoss class and creating test data.
        """
        self.log_loss = LogLoss()
        self.Y = np.array([1, 0, 1, 0])
        self.Yhat = np.array([0.9, 0.1, 0.8, 0.2])
        self.epsilon = 1e-7

    def test_eval(self):
        """
        Test the eval method to ensure it calculates the logarithmic loss correctly.
        """
        calculated_loss = self.log_loss.eval(self.Y, self.Yhat, self.epsilon)
        expected_loss = -np.mean(
            self.Y * np.log(self.Yhat + self.epsilon) +
            (1 - self.Y) * np.log(1 - self.Yhat + self.epsilon)
        )
        self.assertAlmostEqual(calculated_loss, expected_loss, msg="The calculated loss does not match the expected loss.")

    def test_gradient(self):
        """
        Test the gradient method to ensure it calculates the gradient of the loss correctly.
        """
        calculated_gradient = self.log_loss.gradient(self.Y, self.Yhat, self.epsilon)
        expected_gradient = -((self.Y - self.Yhat) / (self.Yhat * (1 - self.Yhat) + self.epsilon))
        np.testing.assert_array_almost_equal(calculated_gradient, expected_gradient, err_msg="The calculated gradient does not match the expected gradient.")

if __name__ == '__main__':
    unittest.main()
