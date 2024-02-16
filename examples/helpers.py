# Imports
import numpy as np
from layers import FullyConnectedLayer



def smape( y_true, y_pred ):
	"""
	Calculate the Symmetric Mean Absolute Percentage Error (SMAPE) between two arrays.

	:param y_true: The ground truth values.
	:param y_pred: The predicted values.
	:rtype: float

	:return: The SMAPE score.
	"""

	# Calculate the denominator for SMAPE, avoiding division by zero
	denominator = np.abs(y_true) + np.abs(y_pred)

	# Compute SMAPE
	return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / denominator, where = denominator != 0)


def fProp( layers, x, y ):
	"""
	Perform forward propagation through a list of layers.

	:param layers: A list of layers through which to propagate.
	:param x: The input features for propagation.
	:param y: The ground truth labels for loss calculation.
	:rtype: tuple

	:return: A tuple containing the final layer activations and the loss.
	"""

	# Initialize activation with input
	activation = x

	# Propagate through all layers except the last
	for layer in layers[ :-1 ]:
		activation = layer.forward(activation)

	# Calculate loss using the last layer
	loss = layers[ -1 ].eval(y, activation)
	return activation, loss


def bProp( layers, Y, h ):
	"""
	Perform backward propagation through a list of layers for gradient calculation and weight update.

	:param layers: A list of layers through which to propagate backwards.
	:param Y: The ground truth labels.
	:param h: The final layer activations from forward propagation.
	:rtype: np.ndarray

	:return: The gradient calculated after backpropagation through all applicable layers.
	"""

	# Calculate initial gradient based on the loss
	grad = layers[ -1 ].gradient(Y, h)

	# Iterate backwards through layers, skipping the input layer
	for i in range(len(layers) - 2, 0, -1):

		# Calculate new gradient for the current layer
		newgrad = layers[ i ].backward(grad)

		# Update weights if the layer is a FullyConnectedLayer
		if isinstance(layers[ i ], FullyConnectedLayer.FullyConnectedLayer):
			layers[ i ].updateWeights(grad, 1e-4)

		# Update grad for the next iteration
		grad = newgrad

	return grad
