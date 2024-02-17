# Imports
from layers import InputLayer, FullyConnectedLayer2, SoftmaxLayer, CrossEntropy
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from examples.helpers import fProp, bProp

def train_network(layers, xtrain, ytrain, xtest, ytest, epochs=10000, learning_rate=0.001):
    """
    Trains the neural network using forward and backward propagation.

    :param layers: A list containing the layers of the network.
    :param xtrain: Training data features.
    :param ytrain: Training data labels.
    :param xtest: Test data features.
    :param ytest: Test data labels.
    :param epochs: Number of epochs to train for.
    :param learning_rate: Learning rate for weight updates.
    :return: Lists of training and test loss values over epochs.
    """
    jtrain, jtest = [], []
    prevtestce, prevtraince = 0, 0
    for epoch in range(epochs):
        if epoch % 500 == 0:
            print("Epoch:", epoch)

        # Forward and backward propagation on the training set
        sm_forward, ce_forward = fProp(layers, xtrain, ytrain)
        jtrain.append(ce_forward)

        # Backpropagation and weights update
        ce_back = layers[-1].gradient(ytrain, sm_forward)
        sm_back = layers[-2].backward(ce_back)
        layers[1].updateWeights(sm_back, epoch, learning_rate)

        # Forward propagation on the test set
        smtest_forward, cetest_forward = fProp(layers, xtest, ytest)
        jtest.append(cetest_forward)

        # Early stopping check
        if epoch > 0 and cetest_forward > prevtestce and np.abs(prevtraince - ce_forward) < 1e-5:
            print(f"Stopping early at epoch {epoch} due to minimal loss improvement.")
            break
        prevtestce, prevtraince = cetest_forward, ce_forward

    return jtrain, jtest

def plot_metrics(jtrain, jtest):
    """
    Plots the training and test loss over epochs.

    :param jtrain: List of training loss values.
    :param jtest: List of test loss values.
    """
    epochs = range(1, len(jtrain) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, jtrain, label='Training Loss')
    plt.plot(epochs, jtest, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('../img/SoftCEMN_1.png')
    plt.show()

def main():
    """
    Main function to load the dataset, preprocess data, initialize the network,
    train the network, and plot the training and test metrics.
    """
    # Load and preprocess the dataset
    train_df = pd.read_csv("../datasets/mnist_train_100.csv")
    test_df = pd.read_csv("../datasets/mnist_valid_10.csv")
    np.random.seed(0)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    ytrain = train_df.iloc[:, 0].values.reshape(-1, 1)
    ytest = test_df.iloc[:, 0].values.reshape(-1, 1)
    xtrain = train_df.iloc[:, 1:].values
    xtest = test_df.iloc[:, 1:].values
    encoder = OneHotEncoder()
    ytrain = encoder.fit_transform(ytrain).toarray()
    ytest = encoder.transform(ytest).toarray()

    # Initialize network layers
    layers = [InputLayer.InputLayer(xtrain), FullyConnectedLayer2.FullyConnectedLayer2(xtrain.shape[1], ytrain.shape[1]), SoftmaxLayer.SoftmaxLayer(), CrossEntropy.CrossEntropy()]

    # Train the network and plot the metrics
    jtrain, jtest = train_network(layers, xtrain, ytrain, xtest, ytest)
    plot_metrics(jtrain, jtest)

if __name__ == "__main__":
    main()
