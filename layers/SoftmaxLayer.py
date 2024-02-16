# Imports
from layers.Layer import Layer
import numpy as np
import pandas as pd

class SoftmaxLayer(Layer):
    """
    Softmax Layer class.

    This class represents the softmax layer in a neural network, which computes
    the softmax activation function over the input data.

    Methods:
        __init__: Initialize the softmax layer.
        forward: Perform the forward pass through the softmax layer.
        gradient: Compute the gradient of the softmax layer.
        backward: Perform the backward pass through the softmax layer.
    """

    def __init__(self):
        """
        Initialize the softmax layer.
        """
        super().__init__()

    def forward(self, dataIn):
        """
        Perform the forward pass through the softmax layer.

        :param dataIn: Input data as an NxD matrix.
        :type dataIn: np.ndarray or pd.DataFrame
        :return: Output of the softmax layer.
        :rtype: np.ndarray
        """
        # Convert to numpy array if input is a pandas DataFrame
        if isinstance(dataIn, pd.DataFrame):
            dataIn = dataIn.values

        # Set previous input
        self.setPrevIn(dataIn)

        # Compute softmax activation function
        numer = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))
        denom = np.sum(numer, axis=1, keepdims=True)
        y = numer / denom

        # Set previous output
        self.setPrevOut(y)
        return y

    def gradient(self):
        """
        Compute the gradient of the softmax layer.

        :return: Gradient of the softmax layer.
        :rtype: np.ndarray
        """
        # Compute gradient of softmax layer
        y = self.getPrevOut()
        return np.array([np.diag(y_i) - np.outer(y_i, y_i) for y_i in y])

    def backward(self, gradIn):
        """
        Perform the backward pass through the softmax layer.

        :param gradIn: Gradient of the loss with respect to the output of the subsequent layer.
        :type gradIn: np.ndarray
        :return: Gradient of the loss with respect to the input of the softmax layer.
        :rtype: np.ndarray
        """
        # Compute gradient of loss with respect to input of softmax layer
        return np.einsum('...i,...ij', gradIn, self.gradient())
