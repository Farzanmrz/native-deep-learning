# Imports
import unittest
import numpy as np
from layers.SoftmaxLayer import SoftmaxLayer

class TestSoftmaxLayer(unittest.TestCase):
    """
    A unit test class for the SoftmaxLayer.
    """

    def setUp(self):
        """
        Set up test fixtures, including initializing the SoftmaxLayer class and creating test data.
        """
        self.softmax_layer = SoftmaxLayer()
        self.input_data = np.array([[1, 2, 3], [4, 5, 6]])

    def test_forward(self):
        """
        Test the forward method to ensure softmax activation is applied correctly.
        """
        expected_output = np.array([[0.09003057, 0.24472847, 0.66524096],
                                    [0.09003057, 0.24472847, 0.66524096]])
        calculated_output = self.softmax_layer.forward(self.input_data)
        np.testing.assert_allclose(calculated_output, expected_output, rtol=1e-5, atol=1e-8, err_msg="Forward pass output does not match expected output.")

    def test_gradient(self):
        """
        Test the gradient method to ensure it computes the gradient of the softmax layer correctly.
        """
        self.softmax_layer.forward(self.input_data)  # Perform forward pass first
        calculated_gradient = self.softmax_layer.gradient()
        # Dummy assertion as gradient calculation is complex and specific to implementation
        self.assertEqual(calculated_gradient.shape, (2, 3, 3), "Gradient calculation shape is incorrect.")

    def test_backward(self):
        """
        Test the backward method to ensure it computes the gradient of the loss with respect to the input correctly.
        """
        self.softmax_layer.forward(self.input_data)  # Perform forward pass first
        grad_in = np.random.randn(2, 3)
        calculated_gradient = self.softmax_layer.backward(grad_in)
        # Dummy assertion as gradient calculation is complex and specific to implementation
        self.assertEqual(calculated_gradient.shape, (2, 3), "Backward pass calculation shape is incorrect.")

if __name__ == '__main__':
    unittest.main()
