import numpy as np
import random

def genetic_algorithm(x):
    return 0.2 * (x ** 0.5) + 2 * np.sin(2 * np.pi * 0.02 * x) + 5


def random_binary_values_8(count):
    binary_list = []
    for i in range(count):
        binary_list.append(to_binary(random.randint(0, 255), 8))
    return binary_list


def to_binary(int_number, number_of_bits):
    return format(int_number, 'b').zfill(number_of_bits)


def adaptation_indicator(element, sum_of_elements):
    return (element / sum_of_elements) * 100


def to_int(x):
    return int(x, 2)


def roulette(number_of_population, wages_list):
    number_of_children_list = np.zeros(number_of_population)

    for i in range(number_of_population):
        random_number = random.uniform(0, number_of_population)
        sum_of_wages = 0

        for index, value in enumerate(wages_list):
            sum_of_wages += value

            if sum_of_wages > random_number:
                number_of_children_list[index] += 1
                sum_of_wages = 0
    return number_of_children_list


def main():
    pk = 0
    pm = 0
    n = 100
    binary_list = random_binary_values_8(n)
    integer_list = list(map(to_int, binary_list))
    f_value_list = list(map(genetic_algorithm, integer_list))
    sum_of_f_values = sum(f_value_list)
    adaptation_indicator_list = list(map(lambda x: adaptation_indicator(x, sum_of_f_values), f_value_list))
    number_of_children_list = roulette(n, adaptation_indicator_list)
    print(number_of_children_list)


if __name__ == '__main__':
    main()
