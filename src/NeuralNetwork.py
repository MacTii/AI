import random

import numpy
import numpy as np
import matplotlib.pyplot as plt

# treningowe dane dla punktów powyżej y=x i poniżej y=x
training_points_positive = np.array([[-2, 2], [3, 5], [0, 5], [-1, 5], [1, 3]], dtype=float)
training_points_negative = np.array([[4, -2], [4, -3], [2, -4], [3, -1], [2, -2]], dtype=float)
all_training_points = np.concatenate((training_points_positive, training_points_negative))
labelled_data = list(zip(all_training_points,
                         [0] * len(training_points_positive) + [1] * len(training_points_negative)))
random.shuffle(labelled_data)

# losowe dane
point_x = np.random.randint(-10, 10, 50)
point_y = np.random.randint(-10, 10, 50)
points = np.dstack((point_x, point_y))
points.dtype = float


def train(inputs, trainer):
    weights = np.random.rand(1, 2)
    bias = np.random.rand(1, 1)

    everything_right = False
    epochs = 0

    while not everything_right:
        everything_right = True

        for idx, value in enumerate(trainer):

            a = np.dot(weights, inputs[idx])
            a = 1 if a > 0 else 0
            error = value - a

            if error != 0:
                everything_right = False

                delta_weight = error * inputs[idx]
                delta_bias = error

                weights = np.add(weights, delta_weight)
                bias = np.add(bias, delta_bias)
                epochs = epochs + 1

            print(f"Epoch {epochs} (Weight: {weights} Bias: {bias})", weights, bias)

    return weights, bias


if __name__ == '__main__':
    figure, axis = plt.subplots(1, 2)

    # training
    axis[0].plot(training_points_positive[:, 0], training_points_positive[:, 1], 'ro')
    axis[0].plot(training_points_negative[:, 0], training_points_negative[:, 1], 'bo')

    training_points, expected_result = map(list, zip(*labelled_data))
    weights, bias = train(training_points, expected_result)

    print(f"Calculated Weights: {weights} and Bias: {bias}")

    x = np.arange(-10, 10)
    m = -weights[0, 0] / weights[0, 1]
    axis[0].plot(x, m * x, "b", label="Prosta separująca")
    axis[0].legend()

    axis[0].set_title("Zbiór treningowy")

    # data without trainer
    axis[1].plot(point_x, point_y, "o")
    axis[1].plot(x, m * x, "b", label="Prosta separująca")
    axis[1].plot(x, x, "g", label="Oczekiwana prosta")
    axis[1].set_title("Zbiór testu")
    axis[1].legend()

    plt.grid()
    plt.show()