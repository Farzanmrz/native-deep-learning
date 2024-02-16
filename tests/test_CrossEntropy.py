# Imports
import unittest
import numpy as np
from layers.CrossEntropy import CrossEntropy


class TestCrossEntropy(unittest.TestCase):
	"""
	A test case class for the CrossEntropy loss function using the unittest framework.

	Methods:
		test_eval: Tests the eval method for correctness.
		test_gradient: Tests the gradient method for correctness.
		test_eval_zero_division: Tests the eval method to ensure it handles division by zero.
		test_gradient_zero_division: Tests the gradient method to ensure it handles division by zero.
	"""

	def setUp( self ):
		"""
		Set up method to prepare the test fixture. This method is called before each test.
		"""
		self.cross_entropy = CrossEntropy()
		self.epsilon = 1e-7

	def test_eval( self ):
		"""
		Test the eval method to ensure it correctly calculates the cross entropy loss.
		"""
		Y = np.array([ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ])
		Yhat = np.array([ [ 0.7, 0.2, 0.1 ], [ 0.1, 0.8, 0.1 ], [ 0.1, 0.2, 0.7 ] ])
		loss = self.cross_entropy.eval(Y, Yhat)
		expected_loss = -np.mean(np.sum(Y * np.log(Yhat + self.epsilon), axis = 1))
		self.assertAlmostEqual(loss, expected_loss, "Eval method should calculate correct cross entropy loss.")

	def test_gradient( self ):
		"""
		Test the gradient method to ensure it correctly calculates the gradient of the cross entropy loss.
		"""
		Y = np.array([ [ 1, 0, 0 ] ])
		Yhat = np.array([ [ 0.7, 0.2, 0.1 ] ])
		grad = self.cross_entropy.gradient(Y, Yhat)
		expected_grad = -((Y) / (Yhat + self.epsilon))
		np.testing.assert_array_almost_equal(grad, expected_grad, err_msg = "Gradient method should calculate the correct gradient.")

	def test_eval_zero_division( self ):
		"""
		Test the eval method with inputs that would cause a division by zero to ensure it handles it using epsilon.
		"""
		Y = np.array([ [ 1, 0, 0 ] ])
		Yhat = np.array([ [ 0, 1, 0 ] ])  # This would cause a division by zero if not for epsilon
		loss = self.cross_entropy.eval(Y, Yhat)
		self.assertFalse(np.isnan(loss), "Eval method should not result in NaN even when division by zero occurs.")

	def test_gradient_zero_division( self ):
		"""
		Test the gradient method with inputs that would cause a division by zero to ensure it handles it using epsilon.
		"""
		Y = np.array([ [ 1, 0, 0 ] ])
		Yhat = np.array([ [ 0, 1, 0 ] ])  # This would cause a division by zero if not for epsilon
		grad = self.cross_entropy.gradient(Y, Yhat)
		self.assertFalse(np.isnan(grad).any(), "Gradient method should not result in NaN even when division by zero occurs.")


if __name__ == '__main__':
	unittest.main()
