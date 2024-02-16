# Imports
import unittest
import numpy as np
from layers.FullyConnectedLayer import FullyConnectedLayer


class TestFullyConnectedLayer(unittest.TestCase):
	"""
	A test case class for the FullyConnectedLayer using the unittest framework.

	Methods:
		setUp: Set up method to prepare the test fixture before each test method is called.
		test_initialization: Tests the __init__ method to ensure weights and biases are initialized correctly.
		test_weight_bias_getters: Tests the getWeights and getBiases methods.
		test_weight_bias_setters: Tests the setWeights and setBiases methods.
		test_forward: Tests the forward method to ensure proper forward pass.
		test_gradient: Tests the gradient method for correct gradient computation.
		test_backward: Tests the backward method to ensure proper backward pass.
		test_updateWeights: Tests the updateWeights method to ensure weights are updated correctly.
	"""

	def setUp( self ):
		"""
		Set up method to prepare the test fixture. This method is called before each test method.
		"""
		self.sizeIn = 5
		self.sizeOut = 3
		self.fc_layer = FullyConnectedLayer(self.sizeIn, self.sizeOut)

	def test_initialization( self ):
		"""
		Test the __init__ method to ensure weights and biases are initialized with the correct shape.
		"""
		self.assertEqual(self.fc_layer.weights.shape, (self.sizeIn, self.sizeOut))
		self.assertEqual(self.fc_layer.bias.shape, (1, self.sizeOut))

	def test_weight_bias_getters( self ):
		"""
		Test the getWeights and getBiases methods to ensure they return the correct values.
		"""
		weights = self.fc_layer.getWeights()
		biases = self.fc_layer.getBiases()
		self.assertIsInstance(weights, np.ndarray)
		self.assertIsInstance(biases, np.ndarray)

	def test_weight_bias_setters( self ):
		"""
		Test the setWeights and setBiases methods to ensure they properly update the weights and biases.
		"""
		new_weights = np.random.uniform(-1e-4, 1e-4, (self.sizeIn, self.sizeOut))
		new_biases = np.random.uniform(-1e-4, 1e-4, (1, self.sizeOut))
		self.fc_layer.setWeights(new_weights)
		self.fc_layer.setBiases(new_biases)
		np.testing.assert_array_equal(self.fc_layer.weights, new_weights)
		np.testing.assert_array_equal(self.fc_layer.bias, new_biases)

	def test_forward( self ):
		"""
		Test the forward method to ensure it performs a correct forward pass.
		"""
		input_data = np.random.randn(10, self.sizeIn)
		output_data = self.fc_layer.forward(input_data)
		expected_output_shape = (10, self.sizeOut)
		self.assertEqual(output_data.shape, expected_output_shape)

	def test_gradient( self ):
		"""
		Test the gradient method to ensure it returns the correct gradient shape.
		"""
		gradient_matrix = self.fc_layer.gradient()
		self.assertEqual(gradient_matrix.shape, (self.sizeOut, self.sizeIn))

	def test_backward( self ):
		"""
		Test the backward method to ensure it computes the correct backward pass.
		"""
		gradIn = np.random.randn(10, self.sizeOut)
		gradOut = self.fc_layer.backward(gradIn)
		self.assertEqual(gradOut.shape, (10, self.sizeIn))

	def test_updateWeights( self ):
		"""
		Test the updateWeights method to ensure it updates weights and biases correctly.
		"""
		input_data = np.random.randn(10, self.sizeIn)  # Generate random input data
		gradIn = np.random.randn(10, self.sizeOut)  # Generate random gradient data
		self.fc_layer.forward(input_data)  # Perform a forward pass to set _prevIn

		old_weights = self.fc_layer.getWeights().copy()
		old_biases = self.fc_layer.getBiases().copy()
		self.fc_layer.updateWeights(gradIn)  # Now _prevIn is set, and this should work
		new_weights = self.fc_layer.getWeights()
		new_biases = self.fc_layer.getBiases()

		# Assert that weights and biases have been updated (i.e., changed from the old values)
		self.assertFalse(np.array_equal(new_weights, old_weights))
		self.assertFalse(np.array_equal(new_biases, old_biases))


if __name__ == '__main__':
	unittest.main()
