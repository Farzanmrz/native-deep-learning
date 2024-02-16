# Deep Learning Native Implementation

## Project Overview
This project contains a native implementation of deep learning algorithms and examples of shallow networks being built and analyzed for various purposes. The goal is to provide a hands-on experience with the fundamentals of neural network layers and operations without the use of high-level libraries such as TensorFlow or PyTorch.

## Structure
- `main.py`: Main file to run examples and analyses.
- `/layers`: Contains the implementation of input, fully connected, activation, and objective layers.
- `/datasets`: Includes datasets for various analyses.

## Layers
The layers are built on an abstract class `Layer.py` that defines a general architecture with `forward`, `backward`, and `gradient` methods.

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

## Testing
- `/tests`: Contains unit tests for each layer which can be run individually to validate functionality.

## Usage
Instructions on how to run the main analysis files and use the implemented layers will be provided as the project evolves.

## Future Work
- Implement more complex network architectures.
- Include more examples and tutorials on how to use the network for classification, recommendation, and other tasks.
- Enhance the library with additional features and optimizations.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

* Farzan Mirza: farzan.mirza@drexel.edu (https://www.linkedin.com/in/farzan-mirza13/) 


