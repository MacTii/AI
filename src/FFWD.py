import math

import neurolab as nl
import numpy as np
import pylab as pl
from neurolab.error import MSE


def first_example():
    # Tworzymy zbiór ucząc
    x = np.linspace(-7, 7, 30)
    y1 = 2 * x * np.cos(x)
    wsp = np.abs(max(y1) - min(y1))
    y = y1 / wsp
    size = len(x)
    print(y)
    inp = x.reshape(size, 1)
    # print(inp)
    tar = y.reshape(size, 1)
    # Tworzymy sieć z dwoma warstwami, inicjalizowaną w sposób losowy,
    # w pierwszej warstwie x neuronów, w drugiej warstwie 1 neuron
    net = nl.net.newff([[-7, 7]], [10, 1])
    # Uczymy sieć, wykorzystujemy metodę największego spadku gradientu
    # net.trainf = nl.train.train_gd
    net.trainf = nl.train.train_gdx
    error = net.train(inp, tar, epochs=1000, show=100, goal=0.05)
    # error = net.train(x, y, epochs=1000, show=100, goal=0.01)
    # Symulujemy
    out = net.sim(inp)
    print(out)
    # Tworzymy wykres z wynikami
    x2 = np.linspace(-6.0, 6.0, 150)
    y2 = net.sim(x2.reshape(x2.size, 1)).reshape(x2.size)
    y3 = out.reshape(size)
    pl.plot(x2, y2, '-', x, y, '.', x, y3, 'p')
    pl.legend(['wynik uczenia', 'wartosc rzeczywista'])
    pl.show()
    mean_square_error = MSE()
    print(mean_square_error(y, out))


def sim_neural_network(x_input, y_target, min_x, max_x, train_method, neuron_count, name):
    net = nl.net.newff([[min_x, max_x]], [neuron_count, 1])
    net.trainf = train_method
    errors = net.train(x_input, y_target, epochs=1000, show=0, goal=0.05)
    out = net.sim(x_input)
    error_MSE = sum([pow(i, 2) for i in errors]) / len(errors)

    x2 = np.linspace(min_x, max_x, 150)
    y2 = net.sim(x2.reshape(x2.size, 1)).reshape(x2.size)
    y3 = out.reshape(len(x_input))
    pl.plot(x2, y2, '-', x_input.flatten(), y_target.flatten(), '.', x_input.flatten(), y3, 'p')
    pl.title(name + " - " + str(neuron_count) + " neurons. MSE = " + str(error_MSE))
    pl.legend(['wynik uczenia', 'wartosc rzeczywista'])
    pl.savefig(name + "_" + str(neuron_count) + ".png")
    pl.show()

    return error_MSE


if __name__ == '__main__':
    methods_list = [(nl.train.train_gd, "train_gd"),
                    (nl.train.train_gdm, "train_gdm"),
                    (nl.train.train_gda, "train_gda"),
                    (nl.train.train_gdx, "train_gdx"),
                    (nl.train.train_rprop, "train_rprop")]
    neurons_count_list = [3, 5, 10, 15, 30, 50]

    min_x = 0
    max_x = 9

    x = np.linspace(min_x, max_x, 30)
    y = [2 * math.pow(i, 1. / 3.) * math.sin(i / 10) * math.cos(3 * i) for i in x]
    y = np.array(y)
    y = y / np.abs(max(y) - min(y))

    for method in methods_list:
        print("\n")
        for neuron_count in neurons_count_list:
            error = sim_neural_network(x.reshape(len(x), 1), y.reshape(len(y), 1), min_x, max_x, method[0],
                                       neuron_count, method[1])
            print(str(neuron_count) + ": " + str(error))
