import random

import numpy as np
import math
import matplotlib.pyplot as plt


def calculateValues(x):
    values = []
    for i in x:
        values.append(0.2 * (i ** (1 / 2)) + 2 * math.sin(2 * math.pi * 0.02 * i) + 5)
    return np.array(values)


def showPlot(x, y):
    plt.plot(x, y)
    plt.show()


def generateChromosomes(number):
    return np.random.randint(0, 256, number)


def randomPairs(numbers):
    remaining_numbers = numbers.copy()
    pairs = []

    while len(remaining_numbers) > 0:
        index = np.random.randint(0, len(remaining_numbers))
        number1 = remaining_numbers[index]
        remaining_numbers = np.delete(remaining_numbers, index)

        index = np.random.randint(0, len(remaining_numbers))
        number2 = remaining_numbers[index]
        remaining_numbers = np.delete(remaining_numbers, index)

        pairs.append([number1, number2])

    return pairs


def crossoverPair(pair):
    byte_a = format(pair[0], '08b')
    byte_b = format(pair[1], '08b')

    point = random.randint(1, 7)

    after_byte_a = byte_a[:point] + byte_b[point:]
    after_byte_b = byte_b[:point] + byte_a[point:]

    pair[0] = int(after_byte_a, 2)
    pair[1] = int(after_byte_b, 2)


def mutate(chromosome):
    byte = format(chromosome, '08b')
    byte = list(byte)

    mutation_point = random.randint(0, 7)
    byte[mutation_point] = "0" if byte[mutation_point] == "1" else "1"
    byte = "".join(byte)

    return int(byte, 2)


def geneticGeneration(chromosomes, crossover_probability, mutation_probability):
    calculated_values = calculateValues(chromosomes)

    fp = [(value / np.sum(calculated_values)) * 1 for value in calculated_values]
    fp = np.array(fp)

    chromosomes = np.random.choice(chromosomes, len(chromosomes), p=fp)
    # fp = fp * len(fp)

    pairs = randomPairs(chromosomes)
    for pair in pairs:
        if crossover_probability >= random.random():
            crossoverPair(pair)

    chromosomes = [x for pair in pairs for x in pair]

    for idx, chromosome in enumerate(chromosomes):
        if mutation_probability >= random.random():
            chromosomes[idx] = mutate(chromosome)

    return np.average(calculated_values), chromosomes


def geneticAlgorithm(number_of_generation, chromosomes_number, crossover_probability, mutation_probability):
    fp_s = []
    chromosomes = generateChromosomes(chromosomes_number)

    for idx in range(number_of_generation):
        max_fp, chromosomes = geneticGeneration(chromosomes, crossover_probability, mutation_probability)
        fp_s.append(max_fp)

    return fp_s, chromosomes


arguments = np.linspace(0, 255, 256, dtype=int)
values = calculateValues(arguments)

numberOfChromosomes = [50, 200]
pk_s = [0.5, 0.6, 0.7, 0.8, 1]
pm_s = [0, 0.01, 0.06, 0.1, 0.2, 0.3, 0.5]

if __name__ == '__main__':

    for number in numberOfChromosomes:
        print("Numer chromosomow "+str(number)+":")
        for pk in pk_s:
            plt.axhline(y=9.91, color="r", linestyle="-")
            print("\tpk "+str(pk)+":")
            for pm in pm_s:
                fp_s, chromosomes = geneticAlgorithm(200, number, pk, pm)
                calculated_values = calculateValues(chromosomes)
                print("\t\tpm "+str(pm)+": "+str(round(np.average(calculated_values), 4)))
                fp_arguments = np.linspace(0, len(fp_s), len(fp_s))
                plt.plot(fp_arguments, fp_s,
                         c=np.random.rand(3, ), label="pm=" + str(pm))
            plt.title("pk = "+str(pk)+", number of chromosomes = "+str(number))
            plt.legend()
            plt.savefig("pk"+str(pk)+"chroms"+str(number) + ".png")
            plt.show()

    plt.axhline(y=9.91, color="r", linestyle="-")
    for pk in pk_s:
        fp_s, chromosomes = geneticAlgorithm(200, 50, pk, pm_s[1])
        fp_arguments = np.linspace(0, len(fp_s), len(fp_s))
        plt.plot(fp_arguments, fp_s,
                 c=np.random.rand(3, ), label="pk=" + str(pk))
    plt.title("pm = " + str(pm_s[1]) + ", number of chromosomes = " + str(50))
    plt.legend()
    plt.savefig("pm" + str(pm_s[1]) + "chroms" + str(50) + ".png")
    plt.show()

    # showPlot(np.linspace(0, len(fp_s), len(fp_s)), fp_s)
    #
    # calculated_values = calculateValues(chromosomes)
    # results = zip(chromosomes, calculated_values)
    # the_best = max(results, key=lambda x: x[1])
    #
    # plt.plot(arguments, values)
    # plt.plot(the_best[0], the_best[1], marker="o", markersize=20, markeredgecolor="red",
    #          markerfacecolor="green")
    # plt.show()