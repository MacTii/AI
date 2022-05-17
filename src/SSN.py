import random

import numpy as np

np.random.seed(0)

animals = np.array([[4, 2, -1],
                    [0.01, -1, 3.5],
                    [0.01, 2.0, 0.01],
                    [-1, 2.5, -2],
                    [-1.5, 2, 1.5]],
                   dtype=float)
animals_trainer = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                           dtype=float)


class Layer:
    def __init__(self, number_of_inputs, number_of_neurons, unipolar_factor=5, train_factor=0.1):
        self.weights = 0.1 * np.random.randn(number_of_neurons, number_of_inputs)
        self.biases = np.zeros((1, number_of_neurons))

        self.unipolar_factor = unipolar_factor
        self.train_factor = train_factor

        self.output = None
        self.activation_output = None

    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.biases

    def binaryActivation(self, inputs):
        self.activation_output = np.heaviside(inputs, 0) * self.unipolar_factor

    def sigmoidActivation(self, inputs):
        self.activation_output = (1 / (1 + np.exp(-(self.unipolar_factor * inputs))))


def train(layer, inputs, trainer, number_of_epochs, activation_function):
    for i in range(number_of_epochs):
        random_index = random.randint(0, inputs.shape[1] - 1)
        random_input = inputs[:, random_index]
        random_trainer = trainer[:, random_index]

        layer.forward(random_input)
        activation_function(layer.output)

        errors = random_trainer - layer.activation_output
        delta_weights = np.array([layer.train_factor * error * random_input for error in errors[0, :]])
        layer.weights = layer.weights + delta_weights

if __name__ == "__main__":
    neuralLayer = Layer(5, 3)
    train(neuralLayer, animals, animals_trainer, 10, neuralLayer.sigmoidActivation)

    results = np.empty((0, 3), dtype=float)

    for column in animals.transpose():
        neuralLayer.forward(column.transpose())
        neuralLayer.sigmoidActivation(neuralLayer.output)
        output = np.array(neuralLayer.activation_output)
        results = np.append(results, output, axis=0)

    print("Wyniki dla zbioru treningowego: ")
    print(results)

    dolphin = np.array([[-1], [1.5], [-1], [-2], [-1.5]])
    neuralLayer.forward(dolphin)
    neuralLayer.sigmoidActivation(neuralLayer.output)

    dolphinResults = neuralLayer.activation_output
    print("Czy to ssak? -", dolphinResults[0, 0])
    print("Czy to ptak? -", dolphinResults[1, 0])
    print("Czy to ryba? -", dolphinResults[2, 0])