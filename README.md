## Project Overview
This project contains a native implementation of deep learning algorithms and examples of shallow networks being built and analyzed for various purposes. The goal is to provide a hands-on experience with the fundamentals of neural network layers and operations without the use of high-level libraries such as TensorFlow or PyTorch.

## Structure
- `/layers`: Contains the implementation of input, fully connected, activation, and objective layers.
- `/tests`: Contains unit tests for each layer which can be run individually to validate functionality.
- `/datasets`: Includes datasets for various analyses.

  - [README.md](datasets/README.md): Contains detailed information of each dataset used in this project
    
- `/examples`: Contains example scripts demonstrating the application of neural networks using the layers implemented in this project.

  - [`helpers.py`](examples/helpers.py): Contains common functions used across multiple network examples.

  - Naming convention:
    - `0_linear_medical.py`: Indicates 0 activation layers, linear regression used, and applied to the Medical dataset.
    - `1_logsig_kid.py`: Indicates 1 activation layer of the logistic sigmoid type, used on the Kid Creative dataset.
  The pattern is `<number_of_activation_layers>_<type_of_activation_layer>_<dataset_name>`. Activation layer types are abbreviated as `linear`, `logsig` for logistic sigmoid, etc.

## Layers
The layers are built on an abstract class `Layer.py` that defines a general architecture with `forward`, `backward`, and `gradient` methods. Each type of layer is specialized for its role within a neural network.

### Input Layer
The `InputLayer` is responsible for preprocessing the input data by normalizing it. This involves adjusting the data to have a mean of zero and a standard deviation of one, commonly known as z-scoring. This process helps with the convergence of the neural network during training.

### Fully Connected Layer
The `FullyConnectedLayer` is a standard layer in neural networks where every input neuron is connected to every output neuron. It transforms the input data into output data through learned weights and biases. The transformation is a linear operation—weights are multiplied with the input data, and biases are added to these results.

### Activation Functions
Implemented activation functions include:
- **Linear**: For linear transformations.
- **ReLU**: Rectified Linear Unit for non-linear activations.
- **LogisticSigmoid**: Sigmoid activation function for binary classification tasks.
- **Softmax**: For multi-class classification problems.
- **Tanh**: Hyperbolic tangent function for scaled activations.

### Objective Functions
Objective functions implemented for network evaluation include:
- **LogLoss**: For binary classification.
- **CrossEntropy**: For multi-class classification.
- **SquaredError**: For regression tasks.

## Future Work
- Implement more complex network architectures.
- Include more examples and tutorials on how to use the network for classification, recommendation, and other tasks.
- Enhance the library with additional features and optimizations.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
* Farzan Mirza: [farzan.mirza@drexel.edu](mailto:farzan.mirza@drexel.edu) | [LinkedIn](https://www.linkedin.com/in/farzan-mirza13/)
