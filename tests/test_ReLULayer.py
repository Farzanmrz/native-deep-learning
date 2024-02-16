# Imports
import unittest
import numpy as np
from layers.ReLULayer import ReLULayer


class TestReLULayer(unittest.TestCase):
	"""
	A unit test class for the ReLULayer.
	"""

	def setUp( self ):
		"""
		Set up test fixtures, including initializing the ReLULayer class and creating test data.
		"""
		self.relu_layer = ReLULayer()
		self.input_data = np.array([ [ 1, -2, 3 ], [ -4, 5, -6 ] ])

	def test_forward( self ):
		"""
		Test the forward method to ensure ReLU activation is applied correctly.
		"""
		expected_output = np.array([ [ 1, 0, 3 ], [ 0, 5, 0 ] ])
		calculated_output = self.relu_layer.forward(self.input_data)
		np.testing.assert_array_equal(calculated_output, expected_output, err_msg = "Forward pass output does not match expected output.")

	def test_gradient( self ):
		"""
		Test the gradient method to ensure it computes the gradient of ReLU activation correctly.
		"""
		self.relu_layer.forward(self.input_data)  # Perform forward pass first
		calculated_gradient = self.relu_layer.gradient()
		expected_gradient = np.array([ [ 1, 0, 1 ], [ 0, 1, 0 ] ])
		np.testing.assert_array_equal(calculated_gradient, expected_gradient, err_msg = "Gradient calculation is incorrect.")

	def test_backward( self ):
		"""
		Test the backward method to ensure it computes the gradient of the loss with respect to the input correctly.
		"""
		self.relu_layer.forward(self.input_data)  # Perform forward pass first
		grad_in = np.array([ [ 1, 2, 3 ], [ 4, 5, 6 ] ])
		calculated_gradient = self.relu_layer.backward(grad_in)
		expected_gradient = np.array([ [ 1, 0, 3 ], [ 0, 5, 0 ] ])
		np.testing.assert_array_equal(calculated_gradient, expected_gradient, err_msg = "Backward pass calculation is incorrect.")


if __name__ == '__main__':
	unittest.main()
