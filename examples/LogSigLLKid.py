# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from layers import InputLayer, FullyConnectedLayer, LogisticSigmoidLayer, LogLoss
from examples.helpers import fProp, bProp


def calculate_accuracy( pred, true ):
	"""
	Calculate the accuracy of predictions against true labels using a threshold of 0.5.

	:param pred: The predicted values, as a numpy array.
	:param true: The true labels, as a numpy array.
	:return: The accuracy as a percentage.
	"""
	correct_predictions = np.sum((pred > 0.5) == true)
	return correct_predictions / len(true) * 100.0


def train_network( layers, xtrain, ytrain, xval, yval, epochs ):
	"""
	Trains a neural network by performing forward and backward propagation across specified layers.

	:param layers: A list of layer instances forming the neural network.
	:param xtrain: Training set features.
	:param ytrain: Training set labels.
	:param xval: Validation set features.
	:param yval: Validation set labels.
	:param epochs: The number of training iterations.
	:return: A tuple containing arrays of training and validation log losses for each epoch, and the final training and validation accuracy.
	"""
	lltrain, llval = [ ], [ ]
	prevll = float('inf')  # Initialize previous log loss with a high value for comparison

	for epoch in range(epochs):
		# Forward and backward propagation on the training data
		train_pred, train_loss = fProp(layers, xtrain, ytrain)
		lltrain.append(train_loss)
		bProp(layers, ytrain, train_pred)

		# Forward propagation on the validation data
		val_pred, val_loss = fProp(layers, xval, yval)
		llval.append(val_loss)

		# Check for early stopping based on minimal change in log loss
		if epoch == epochs - 1 or np.abs(train_loss - prevll) < 1e-10:
			break

		prevll = train_loss

	# Calculate training and validation accuracy
	train_acc = calculate_accuracy(train_pred, ytrain)
	val_acc = calculate_accuracy(val_pred, yval)

	return lltrain, llval, train_acc, val_acc


def plot_metrics( lltrain, llval ):
	"""
	Plots the training and validation log loss across epochs.

	:param lltrain: An array of training log loss values per epoch.
	:param llval: An array of validation log loss values per epoch.
	"""
	epochs = range(1, len(lltrain) + 1)
	plt.figure(figsize = (10, 5))
	plt.plot(epochs, lltrain, label = 'Training Log Loss')
	plt.plot(epochs, llval, label = 'Validation Log Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Log Loss')
	plt.title('Training and Validation Log Loss vs Epoch')
	plt.legend()
	plt.grid(True)
	plt.savefig('../img/LogSigLLKid_1.png')
	plt.show()


def main():
	"""
	Main execution function for neural network training and evaluation.
	"""
	# Data loading and preprocessing
	df = pd.read_csv("../datasets/KidCreative.csv").sample(frac = 1, random_state = 0).reset_index(drop = True)
	X, Y = df.drop('Buy', axis = 1).values, df[ 'Buy' ].values.reshape(-1, 1)

	# Split dataset into training and validation sets
	xtrain, xval, ytrain, yval = train_test_split(X, Y, train_size = 2 / 3, random_state = 0)

	# Define neural network architecture
	layers = [ InputLayer.InputLayer(xtrain), FullyConnectedLayer.FullyConnectedLayer(xtrain.shape[ 1 ], 1), LogisticSigmoidLayer.LogisticSigmoidLayer(),
	           LogLoss.LogLoss() ]

	# Train the network and evaluate performance
	lltrain, llval, train_acc, val_acc = train_network(layers, xtrain, ytrain, xval, yval, 100000)
	plot_metrics(lltrain, llval)

	# Output accuracy results
	print(f'Training Accuracy: {train_acc:.2f}%')
	print(f'Validation Accuracy: {val_acc:.2f}%')


if __name__ == "__main__":
	main()
