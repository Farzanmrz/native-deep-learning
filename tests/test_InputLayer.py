# Imports
import unittest
import numpy as np
from layers.InputLayer import InputLayer


class TestInputLayer(unittest.TestCase):
	"""
	A test case class for the InputLayer using the unittest framework.

	Methods:
		setUp: Set up method to prepare the test fixture before each test method is called.
		test_initialization: Tests the __init__ method to ensure mean and standard deviation are computed correctly.
		test_forward: Tests the forward method to ensure input data is normalized correctly.
		test_gradient_placeholder: Tests the gradient method placeholder.
		test_backward_placeholder: Tests the backward method placeholder.
	"""

	def setUp( self ):
		"""
		Set up method to prepare the test fixture. This method is called before each test method.
		"""
		self.dataIn = np.random.randn(100, 5)  # Generate random input data
		self.input_layer = InputLayer(self.dataIn)

	def test_initialization( self ):
		"""
		Test the __init__ method to ensure mean and standard deviation are computed and set correctly.
		"""
		self.assertEqual(self.input_layer.meanX, self.dataIn.mean())
		self.assertEqual(self.input_layer.stdX, self.dataIn.std())
		# Check if stdX is not zero
		self.assertNotEqual(self.input_layer.stdX, 0)

	def test_forward( self ):
		"""
		Test the forward method to ensure it normalizes the input data correctly.
		"""
		normalized_data = self.input_layer.forward(self.dataIn)
		# Check if the normalization is correct
		np.testing.assert_array_almost_equal(normalized_data.mean(), 0, decimal = 1)
		np.testing.assert_array_almost_equal(normalized_data.std(), 1, decimal = 1)

	def test_gradient_placeholder( self ):
		"""
		Test the gradient method to ensure NotImplementedError is raised.
		"""
		# Verify that NotImplementedError is raised when calling gradient
		with self.assertRaises(NotImplementedError):
			self.input_layer.gradient()

	def test_backward_placeholder( self ):
		"""
		Test the backward method to ensure NotImplementedError is raised.
		"""
		gradIn = np.random.randn(100, 5)
		# Verify that NotImplementedError is raised when calling backward
		with self.assertRaises(NotImplementedError):
			self.input_layer.backward(gradIn)


if __name__ == '__main__':
	unittest.main()
