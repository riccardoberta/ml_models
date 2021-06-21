import math
import tqdm
from typing import List
from scratch.gradient_descent import gradient_step
from scratch.linear_algebra import Vector, dot

def step(x: float) -> float:
    return 1.0 if x >= 0 else 0.0

def perceptron(w: Vector, bias: float, x: Vector) -> float:
    """Returns 1 if the perceptron 'fires', 0 if not"""
    z = dot(weights, x) + bias
    return step(z)

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def neuron(w: Vector, x: Vector) -> float:
    # weights includes the bias term, inputs includes a 1
    return sigmoid(dot(w, x))

def feed_forward(neural_network: List[List[Vector]], input_vector: Vector) -> List[Vector]:
    outputs: List[Vector] = []
    for layer in neural_network:
        input_with_bias = input_vector + [1]              
        output = [neuron(n, input_with_bias) for n in layer]               
        outputs.append(output)
        input_vector = output
    return outputs

def sqerror_gradients(network: List[List[Vector]],
                      input_vector: Vector,
                      target_vector: Vector) -> List[List[Vector]]:
    """
    Given a neural network, an input vector, and a target vector,
    make a prediction and compute the gradient of the squared error
    loss with respect to the neuron weights.
    """
    # forward pass
    hidden_outputs, outputs = feed_forward(network, input_vector)

    # gradients with respect to output neuron pre-activation outputs
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, target_vector)]

    # gradients with respect to output neuron weights
    output_grads = [[output_deltas[i] * hidden_output
                     for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]

    # gradients with respect to hidden neuron pre-activation outputs
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                         dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # gradients with respect to hidden neuron weights
    hidden_grads = [[hidden_deltas[i] * input for input in input_vector + [1]]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]

def train(network: List[List[Vector]], xs: Vector, ys: Vector, epochs: int, learning_rate: float) -> List[List[Vector]]:
    for epoch in tqdm.trange(epochs, desc="Neural network for xor"):
        for x, y in zip(xs, ys):
            gradients = sqerror_gradients(network, x, y)
            # Take a gradient step for each neuron in each layer
            network = [[gradient_step(neuron, grad, -learning_rate)
                        for neuron, grad in zip(layer, layer_grad)]
                        for layer, layer_grad in zip(network, gradients)]
    return network;



