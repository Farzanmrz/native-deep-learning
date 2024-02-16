# Imports
from layers import InputLayer, LogisticSigmoidLayer, FullyConnectedLayer, LogLoss
import pandas as pd
import numpy as np


# Read the file create features
df = pd.read_csv("datasets/KidCreative.csv")
df.drop(df.columns[0], axis=1, inplace=True)
Y = df.iloc[:, 0].values.reshape(-1, 1)
X = df.iloc[:, 1:].values


#Given input X
L1 = InputLayer.InputLayer(X)
L2 = FullyConnectedLayer.FullyConnectedLayer(X.shape[1 ], Y.shape[1 ])
L3 = LogisticSigmoidLayer.LogisticSigmoidLayer()
L4 = LogLoss.LogLoss()
layers = [L1,L2,L3,L4]

#forwards!
h = X
for i in range(len(layers)-1):
    h = layers[i].forward(h)


# Set Yhat = h
Yhat = h


# Backward pass
grad_loss = L4.gradient(Y, h)
grad_sigmoid = L3.backward(grad_loss)
grad_fc = L2.backward(grad_sigmoid)

print(grad_loss)
print(grad_sigmoid)
print(grad_fc)

mean_grad_loss = np.mean(grad_loss, axis=0)
mean_grad_sigmoid = np.mean(grad_sigmoid, axis=0)
mean_grad_fc = np.mean(grad_fc, axis=0)


print("Mean gradient of Log Loss layer:", mean_grad_loss)
print("Mean gradient of Sigmoid layer:", mean_grad_sigmoid)
print("Mean gradient of Fully Connected layer:", mean_grad_fc)





