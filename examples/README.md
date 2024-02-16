# Examples Directory

In this directory, you'll find a series of examples illustrating the application of various neural network architectures to different datasets. Each example demonstrates a unique combination of layers and loss functions to model a specific type of dataset. All models follow a general procedure where the dataset is read, shuffled, split into training and validation sets, and then a neural network is trained to minimize the chosen loss function.

**Naming Convention**:
- The first part (e.g., `0_`): Number of activation layers.
- The second part (e.g., `linear_`): Type of regression or classification task.
- The third part (e.g., `medical`): Dataset on which the network is trained.

**General Architecture**: ``Input Layer → Fully Connected Layer → [Optional Activation Layer(s)] → Loss Layer``

## 0_linear_medical.py
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
![Training and Validation MSE Plot](img/0_linear_medical_1.png)


