import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from layers import InputLayer, FullyConnectedLayer, SquaredError
from helpers import smape, fProp, bProp


def train_network( layers, xtrain, ytrain, xval, yval, epochs ):
	msetrain, mseval = [ ], [ ]
	prevmse = float('inf')

	for epoch in range(epochs):
		train_pred, train_loss = fProp(layers, xtrain, ytrain)
		msetrain.append(train_loss)

		bProp(layers, ytrain, train_pred)

		val_pred, val_loss = fProp(layers, xval, yval)
		mseval.append(val_loss)

		if epoch == epochs - 1 or np.abs(train_loss - prevmse) < 1e-10:
			rmsetrain = np.sqrt(train_loss)
			rmseval = np.sqrt(val_loss)
			smapetrain = smape(ytrain, train_pred)
			smapeval = smape(yval, val_pred)

			print(f"Epoch: {epoch + 1}")
			print(f"Final RMSE Training: {rmsetrain}")
			print(f"Final RMSE Validation: {rmseval}")
			print(f"Final SMAPE Training: {smapetrain}")
			print(f"Final SMAPE Validation: {smapeval}")
			break

		prevmse = train_loss

	return msetrain, mseval


def plot_metrics( msetrain, mseval ):
	epochs = range(1, len(msetrain) + 1)
	plt.figure(figsize = (10, 5))
	plt.plot(epochs, msetrain, label = 'Training MSE')
	plt.plot(epochs, mseval, label = 'Validation MSE')
	plt.xlabel('Epochs')
	plt.ylabel('Mean Squared Error')
	plt.title('Training and Validation MSE vs Epoch')
	plt.legend()
	plt.grid(True)
	plt.show()


def main():
	df = pd.read_csv("../datasets/medical.csv")
	df = df.sample(frac = 1, random_state = 0).reset_index(drop = True)
	xtrain, xval, ytrain, yval = train_test_split(df.drop('charges', axis = 1).values, df[ 'charges' ].values.reshape(-1, 1), train_size = 2 / 3, random_state = 0)

	layers = [ InputLayer.InputLayer(xtrain), FullyConnectedLayer.FullyConnectedLayer(xtrain.shape[ 1 ], ytrain.shape[ 1 ]), SquaredError.SquaredError() ]

	msetrain, mseval = train_network(layers, xtrain, ytrain, xval, yval, 100000)
	plot_metrics(msetrain, mseval)


if __name__ == "__main__":
	main()
