# Imports
import unittest
import numpy as np
from layers.LinearLayer import LinearLayer


class TestLinearLayer(unittest.TestCase):
	"""
	A test case class for the LinearLayer using the unittest framework.

	Methods:
		setUp: Set up method to prepare the test fixture before each test method is called.
		test_forward: Tests the forward method to ensure input data is passed through unchanged.
		test_gradient: Tests the gradient method to ensure it returns an array of ones with the same shape as the input.
		test_backward: Tests the backward method to ensure it passes gradients unchanged.
	"""

	def setUp( self ):
		"""
		Set up method to prepare the test fixture. This method is called before each test method.
		"""
		self.linear_layer = LinearLayer()
		self.dataIn = np.random.randn(5, 3)  # Example input data

	def test_forward( self ):
		"""
		Test the forward method to ensure it passes the input data through unchanged.
		"""
		output_data = self.linear_layer.forward(self.dataIn)
		np.testing.assert_array_equal(output_data, self.dataIn, "Forward pass should return the input data unchanged.")

	def test_gradient( self ):
		"""
		Test the gradient method to ensure it returns an array of ones with the same shape as the input.
		"""
		# Perform a forward pass to set the previous output
		self.linear_layer.forward(self.dataIn)
		gradients = self.linear_layer.gradient()
		expected_gradients = np.ones_like(self.dataIn)
		np.testing.assert_array_equal(gradients, expected_gradients, "Gradients should be an array of ones with the same shape as the input.")

	def test_backward( self ):
		"""
		Test the backward method to ensure it passes the input gradients unchanged.
		"""
		gradIn = np.random.randn(5, 3)  # Example gradient input
		output_grad = self.linear_layer.backward(gradIn)
		np.testing.assert_array_equal(output_grad, gradIn, "Backward pass should return the input gradients unchanged.")


if __name__ == '__main__':
	unittest.main()
