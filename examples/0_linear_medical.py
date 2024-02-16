# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from layers import InputLayer, FullyConnectedLayer, SquaredError
from helpers import smape, fProp, bProp


def train_network( layers, xtrain, ytrain, xval, yval, epochs ):
	"""
	Train the neural network by performing forward and backward propagation.

	:param layers: List of layers forming the neural network.
	:param xtrain: Training features.
	:param ytrain: Training labels.
	:param xval: Validation features.
	:param yval: Validation labels.
	:param epochs: Number of epochs to train for.
	:rtype: tuple

	:return: Tuple containing training and validation MSE for each epoch.
	"""
	msetrain, mseval = [ ], [ ]
	prevmse = float('inf')  # Initialize previous MSE with a large number

	for epoch in range(epochs):
		# Forward propagation on training data
		train_pred, train_loss = fProp(layers, xtrain, ytrain)

		# Append training loss to MSE list
		msetrain.append(train_loss)

		# Backward propagation (update weights)
		bProp(layers, ytrain, train_pred)

		# Forward propagation on validation data
		val_pred, val_loss = fProp(layers, xval, yval)

		# Append validation loss to MSE list
		mseval.append(val_loss)

		# Early stopping condition based on change in loss
		if epoch == epochs - 1 or np.abs(train_loss - prevmse) < 1e-10:

			# Calculate root mean square error for the last epoch
			rmsetrain = np.sqrt(train_loss)
			rmseval = np.sqrt(val_loss)

			# Calculate SMAPE for training and validation
			smapetrain = smape(ytrain, train_pred)
			smapeval = smape(yval, val_pred)

			# Print final metrics
			print(f"Epoch: {epoch + 1}")
			print(f"Final RMSE Training: {rmsetrain}")
			print(f"Final RMSE Validation: {rmseval}")
			print(f"Final SMAPE Training: {smapetrain}")
			print(f"Final SMAPE Validation: {smapeval}")

			# Stop training if change in loss is below threshold
			break

		# Update previous MSE
		prevmse = train_loss

	# Return training and validation MSE
	return msetrain, mseval


def plot_metrics( msetrain, mseval ):
	"""
	Plot the training and validation Mean Squared Error over epochs.

	:param msetrain: List of training MSE values.
	:param mseval: List of validation MSE values.
	"""
	epochs = range(1, len(msetrain) + 1)
	plt.figure(figsize = (10, 5))
	plt.plot(epochs, msetrain, label = 'Training MSE')
	plt.plot(epochs, mseval, label = 'Validation MSE')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Squared Error')
	plt.title('Training and Validation MSE vs Epoch')
	plt.legend()
	plt.grid(True)
	plt.savefig('../img/0_linear_medical_1.png')
	plt.show()  # Display the plot


def main():
	"""
	Main function to execute the neural network training and plotting metrics.
	"""
	# Load the dataset and shuffle
	df = pd.read_csv("../datasets/medical.csv")
	df = df.sample(frac = 1, random_state = 0).reset_index(drop = True)

	# Split the data into training and validation sets
	xtrain, xval, ytrain, yval = train_test_split(df.drop('charges', axis = 1).values, df[ 'charges' ].values.reshape(-1, 1), train_size = 2 / 3, random_state = 0)

	# Initialize the neural network layers
	layers = [ InputLayer.InputLayer(xtrain), FullyConnectedLayer.FullyConnectedLayer(xtrain.shape[ 1 ], ytrain.shape[ 1 ]), SquaredError.SquaredError() ]

	# Train the network and plot the metrics
	msetrain, mseval = train_network(layers, xtrain, ytrain, xval, yval, 100000)
	plot_metrics(msetrain, mseval)


if __name__ == "__main__":
	main()
