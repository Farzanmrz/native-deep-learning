# Examples Directory

In this directory, you'll find a series of examples illustrating the application of various neural network architectures to different datasets. Each example demonstrates a unique combination of layers and loss functions to model a specific type of dataset. All models follow a general procedure where the dataset is read, shuffled, split into training and validation sets, and then a neural network is trained to minimize the chosen loss function.

The examples adhere to a naming convention that aids in identifying the model structure and the dataset used:
- The first part (e.g., `Lin`) indicates the model being used for activation
    -   **Lin**: Linear
    -   **Relu**: Rectified Linear Unit
    -   **LogSig**: Logistic Sigmoid
    -   **Soft**: Softmax
    -   **Tanh**: Tanh
 
- The second part (e.g., `SE`) indicates the objective function being used for loss
    -   **SE**: Squared Error
    -   **LL**: Log Loss
    -   **CE**: Cross Entropy

- The third part (e.g., `Med`) specifies the dataset on which the network is trained
    -   **Kid**: Kid Creative
    -   **Med**: Medical Cost Personal
    -   **MN**: MNIST Handwritten Digit


General Architecture: ``Input Layer → Fully Connected Layer → [Optional Activation Layer(s)] → Loss Layer``

## LinSEMed.py
<p align="center">
<code>Input Layer → Fully Connected Layer → Squared Error Objective Layer</code>
</p>

After training a linear regression model on the medical cost dataset with a squared error objective function and no activation layers,
the process completes 100,000 epochs. The final performance, including the Root Mean Square Error (RMSE) and Symmetric Mean Absolute Percentage
Error (SMAPE) for both training and validation, is printed out. These numbers tell us how well the model is doing. A plot showing how the Mean Squared 
Error (MSE) changes over time is also included below, giving a clear picture of the model's performance throughout the training

### Output
```plaintext
Epoch: 100000
Final RMSE Training: 11373.330004372207
Final RMSE Validation: 11064.706089564706
Final SMAPE Training: 71.65843230171302
Final SMAPE Validation: 68.3006723871572
```
<p align="center">
  <img src="../img/LinSEMed_1.png" alt="Training and Validation MSE Plot">
</p>


