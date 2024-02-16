import unittest
from layers.Layer import Layer


class LayerChild(Layer):
	"""
	A concrete implementation of the Layer abstract base class (ABC) for testing purposes.
	The methods 'forward', 'gradient', and 'backward' are required implementations
	for any subclass of Layer. Here, they are implemented with simple placeholder logic
	to enable instantiation and testing of the Layer base class functionality.
	"""

	def forward( self, dataIn ):
		"""
		A placeholder implementation of the forward pass, which simply returns the input data.
		In a real layer, this method would contain the logic for the forward propagation of input data.

		:param dataIn: Input data to the layer.
		:return: The same input data for testing purposes.
		"""
		return dataIn

	def gradient( self ):
		"""
		A placeholder implementation of the gradient computation.
		In a real layer, this method would compute the gradient of the loss with respect to the layer's parameters.
		"""
		pass

	def backward( self, gradIn ):
		"""
		A placeholder implementation of the backward pass.
		In a real layer, this method would perform backpropagation, computing the gradient of the loss with respect to the input.

		:param gradIn: The gradient of the loss with respect to the layer's output.
		"""
		pass


class TestLayer(unittest.TestCase):
	"""
	A test case class that uses the unittest framework to validate the functionality of the Layer base class.

	Methods:
		setUp: A special method run before each test function to set up any objects that may be used in testing.
		test_set_and_get_prev_in: Tests both setting and getting the previous layer's input.
		test_set_and_get_prev_out: Tests both setting and getting the previous layer's output.
		test_forward: Tests the forward method using a dummy implementation.
	"""

	def setUp( self ):
		"""
		Set up method to prepare the test fixture. This method is called before each test.
		"""
		self.layer = LayerChild()

	def test_set_and_get_prev_in( self ):
		"""
		Test the setPrevIn and getPrevIn methods to ensure they accurately set and return the previous layer's input.
		"""
		test_input = [ 1, 2, 3 ]
		self.layer.setPrevIn(test_input)
		self.assertEqual(self.layer.getPrevIn(), test_input, "getPrevIn should return what was set by setPrevIn")

	def test_set_and_get_prev_out( self ):
		"""
		Test the setPrevOut and getPrevOut methods to ensure they accurately set and return the next layer's output.
		"""
		test_output = [ 4, 5, 6 ]
		self.layer.setPrevOut(test_output)
		self.assertEqual(self.layer.getPrevOut(), test_output, "getPrevOut should return what was set by setPrevOut")

	def test_forward( self ):
		"""
		Test the forward method with a dummy implementation.
		Since the dummy forward method returns the input, the test checks if this behavior is consistent.
		"""
		test_input = [ 1, 2, 3 ]
		self.assertEqual(self.layer.forward(test_input), test_input, "forward should return the input data for the dummy implementation")


# Note: Once the gradient and backward methods are implemented in the LayerChild class,  #       corresponding tests should be added here to ensure their correct functionality.


if __name__ == '__main__':
	unittest.main()
