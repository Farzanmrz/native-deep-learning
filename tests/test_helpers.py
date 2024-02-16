import unittest
from unittest.mock import MagicMock
import numpy as np
from examples.helpers import smape, fProp, bProp  # Adjust the import path as necessary

class TestNeuralNetworkFunctions(unittest.TestCase):
    """
    A test class for neural network utility functions including SMAPE, forward propagation, and backward propagation.
    """

    def test_smape(self):
        """
        Tests the SMAPE function to ensure it calculates the Symmetric Mean Absolute Percentage Error accurately.
        """
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        expected_smape = 100 / 4 * (np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))
        result = smape(y_true, y_pred)
        self.assertAlmostEqual(expected_smape, result, places=5)

    def test_fProp(self):
        """
        Tests the forward propagation function (fProp) to ensure it correctly propagates inputs through a mock neural network.
        """
        # Mock layers
        mock_input_layer = MagicMock()
        mock_fc_layer = MagicMock()
        mock_output_layer = MagicMock()

        # Set side effects
        mock_input_layer.forward.return_value = np.array([1, 2])
        mock_fc_layer.forward.return_value = np.array([2, 4])
        mock_output_layer.eval.return_value = 0.5

        layers = [mock_input_layer, mock_fc_layer, mock_output_layer]
        x = np.array([1, 2])
        y = np.array([2, 4])

        activation, loss = fProp(layers, x, y)

        self.assertTrue(np.array_equal(activation, np.array([2, 4])))
        self.assertEqual(loss, 0.5)

        # Check that eval was called correctly
        args, kwargs = mock_output_layer.eval.call_args
        np.testing.assert_array_equal(args[0], y)  # First argument of eval call
        np.testing.assert_array_equal(args[1], np.array([2, 4]))  # Second argument of eval call

    def test_bProp(self):
        """
        Tests the backward propagation function (bProp) to ensure it correctly calculates gradients and updates weights.
        """
        # Mock layers
        mock_fc_layer = MagicMock()
        mock_output_layer = MagicMock()

        # Set side effects for gradient and backward functions
        grad_from_output_layer = np.array([0.1, 0.2])
        grad_for_fc_layer = np.array([0.05, 0.1])
        mock_output_layer.gradient.return_value = grad_from_output_layer
        mock_fc_layer.backward.return_value = grad_for_fc_layer

        layers = [MagicMock(), mock_fc_layer, mock_output_layer]  # Input layer is mocked just for completeness
        Y = np.array([2, 4])
        h = np.array([2, 4])

        # Perform backward propagation
        gradient = bProp(layers, Y, h)

        # Assert the backward method was called with correct gradients
        mock_output_layer.gradient.assert_called_once()
        mock_fc_layer.backward.assert_called_once()

        # Extract the actual arguments used in the backward call
        backward_call_args = mock_fc_layer.backward.call_args[0][0]
        np.testing.assert_array_equal(backward_call_args, grad_from_output_layer)

        # Lastly, check the final gradient returned matches expectation
        np.testing.assert_array_equal(gradient, grad_for_fc_layer)

if __name__ == '__main__':
    unittest.main()
