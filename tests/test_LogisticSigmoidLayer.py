# Imports
import unittest
import numpy as np
from layers.LogisticSigmoidLayer import LogisticSigmoidLayer

class TestLogisticSigmoidLayer(unittest.TestCase):
    """
    A test case class for the LogisticSigmoidLayer using the unittest framework.

    Methods:
        setUp: Set up method to prepare the test fixture before each test method is called.
        test_forward: Tests the forward method to ensure it correctly applies the logistic sigmoid function.
        test_gradient: Tests the gradient method to ensure it correctly computes the derivative of the logistic sigmoid function.
        test_backward: Tests the backward method to ensure it correctly computes the gradient of the loss with respect to the input.
    """

    def setUp(self):
        """
        Set up method to prepare the test fixture. This method is called before each test method.
        """
        self.sigmoid_layer = LogisticSigmoidLayer()
        self.dataIn = np.array([[0.5, -1], [1, -1.5]])  # Example input data

    def test_forward(self):
        """
        Test the forward method to ensure it correctly applies the logistic sigmoid function to the input data.
        """
        output_data = self.sigmoid_layer.forward(self.dataIn)
        expected_output = 1 / (1 + np.exp(-self.dataIn))
        np.testing.assert_array_almost_equal(output_data, expected_output, decimal=6, err_msg="Forward method should correctly apply the logistic sigmoid function.")

    def test_gradient(self):
        """
        Test the gradient method to ensure it correctly computes the derivative of the logistic sigmoid function.
        """
        self.sigmoid_layer.forward(self.dataIn)  # Necessary to set previous output
        gradients = self.sigmoid_layer.gradient()
        y_hat = self.sigmoid_layer.getPrevOut()
        expected_gradients = y_hat * (1 - y_hat)
        np.testing.assert_array_almost_equal(gradients, expected_gradients, decimal=6, err_msg="Gradient method should correctly compute the derivative of the logistic sigmoid function.")

    def test_backward(self):
        """
        Test the backward method to ensure it correctly computes the gradient of the loss with respect to the input.
        """
        gradIn = np.array([[0.1, 0.2], [-0.1, 0.3]])  # Example gradient input
        self.sigmoid_layer.forward(self.dataIn)  # Necessary to compute gradients
        backward_gradient = self.sigmoid_layer.backward(gradIn)
        expected_backward_gradient = self.sigmoid_layer.gradient() * gradIn
        np.testing.assert_array_almost_equal(backward_gradient, expected_backward_gradient, decimal=6, err_msg="Backward method should correctly compute the gradient of the loss with respect to the input.")

if __name__ == '__main__':
    unittest.main()
