import unittest
import numpy as np
from layers.tanhLayer import tanhLayer

class TestTanhLayer(unittest.TestCase):
    """
    A unit test class for the tanhLayer.
    """

    def setUp(self):
        """
        Set up test fixtures, including initializing the tanhLayer class and creating test data.
        """
        self.tanh_layer = tanhLayer()
        self.input_data = np.array([[1, -2, 3], [-4, 5, -6]])

    def test_forward(self):
        """
        Test the forward method to ensure tanh activation is applied correctly.
        """
        expected_output = np.tanh(self.input_data)
        calculated_output = self.tanh_layer.forward(self.input_data)
        np.testing.assert_array_almost_equal(calculated_output, expected_output, err_msg="Forward pass output does not match expected output.")

    def test_gradient(self):
        """
        Test the gradient method to ensure it computes the gradient of the tanh activation correctly.
        """
        self.tanh_layer.forward(self.input_data)  # Perform forward pass first
        calculated_gradient = self.tanh_layer.gradient()
        expected_gradient = 1 - (np.tanh(self.input_data) ** 2)
        np.testing.assert_array_almost_equal(calculated_gradient, expected_gradient, err_msg="Gradient calculation is incorrect.")

    def test_backward(self):
        """
        Test the backward method to ensure it computes the gradient of the loss with respect to the input correctly.
        """
        self.tanh_layer.forward(self.input_data)  # Perform forward pass first
        grad_in = np.random.randn(*self.input_data.shape)
        calculated_gradient = self.tanh_layer.backward(grad_in)
        expected_gradient = grad_in * (1 - (np.tanh(self.input_data) ** 2))
        np.testing.assert_array_almost_equal(calculated_gradient, expected_gradient, err_msg="Backward pass calculation is incorrect.")

if __name__ == '__main__':
    unittest.main()
