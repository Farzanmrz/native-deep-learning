import unittest
from layers import FullyConnectedLayer2
import numpy as np

class TestFullyConnectedLayer2(unittest.TestCase):
    """
    Unit tests for the FullyConnectedLayer2 class from the layers module.

    Tests include initialization, forward pass, backward pass, and weight update functionalities
    to ensure the layer behaves as expected under various conditions.
    """

    def setUp(self):
        """
        Set up function to initialize a FullyConnectedLayer2 instance and sample data before each test.

        This function initializes a FullyConnectedLayer2 layer with a specified input and output size,
        generates a sample input dataset, and prepares a sample gradient for testing the backward pass
        and weight update functionalities.
        """
        # Initialize the layer with a specific input and output size
        self.layer = FullyConnectedLayer2.FullyConnectedLayer2(sizeIn=5, sizeOut=2)
        # Create a sample input data (5 features, 10 samples)
        self.sample_input = np.random.rand(10, 5)
        # Create a sample gradient for testing backward pass and updates
        self.sample_grad = np.random.rand(10, 2)

    def test_initialization(self):
        """
        Test to verify that the FullyConnectedLayer2 instance is initialized with the correct sizes
        for weights and biases.
        """
        self.assertEqual(self.layer.weights.shape, (5, 2), "Weights initialized to incorrect shape.")
        self.assertEqual(self.layer.bias.shape, (1, 2), "Bias initialized to incorrect shape.")

    def test_forward_pass(self):
        """
        Test to ensure the forward pass correctly computes outputs given input data.

        The output shape is validated to ensure it matches the expected dimensions based on
        the layer's output size configuration.
        """
        output = self.layer.forward(self.sample_input)
        self.assertEqual(output.shape, (10, 2), "Forward pass output shape is incorrect.")

    def test_backward_pass(self):
        """
        Test to verify the backward pass computes gradients correctly.

        This test ensures that the backward pass, which calculates the gradient of the loss with respect
        to the input of the layer, returns a tensor of the correct shape.
        """
        # Ensuring gradients can only be calculated after a forward pass
        self.layer.forward(self.sample_input)
        grad_output = self.layer.backward(self.sample_grad)
        self.assertEqual(grad_output.shape, (10, 5), "Backward pass gradient shape is incorrect.")

    def test_update_weights(self):
        """
        Test to check that weights and biases are updated correctly using the ADAM optimization algorithm.

        This method verifies that after performing a weight update, the weights and biases of the layer
        are actually changed, indicating the update operation is functioning as expected.
        """
        initial_weights = np.copy(self.layer.weights)
        # Ensuring a forward and backward pass to compute gradients
        self.layer.forward(self.sample_input)
        self.layer.backward(self.sample_grad)
        # Update weights
        self.layer.updateWeights(self.sample_grad, t=1, eta=0.01)
        # Verifying weights are updated
        self.assertFalse(np.array_equal(initial_weights, self.layer.weights), "Weights were not updated.")

if __name__ == '__main__':
    unittest.main()
