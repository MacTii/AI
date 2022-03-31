from random import randint
from math import sqrt
from itertools import permutations
from sys import maxsize

def brute_force(cities_distances, index):
    vertex_list = [x for x in range(len(cities_distances)) if x is not index]
    city_permutation = permutations(vertex_list)
    path = []

    for cities in city_permutation:
        path_len = 0
        temp = index
        for city in cities:
            path_len += cities_distances[temp][city]
            temp = city

        path_len += cities_distances[temp][index]
        path.append(round(path_len, 2))

    return min(path)

def city_coordinates(C):
    cityCoord = []
    for i in range(C):
        coord = (randint(0, 100), randint(0, 100))
        if coord not in cityCoord:
            cityCoord.append(coord)

    return cityCoord

def euclidean_distance(l_cityCoord, index):
    eucDist = []
    for i in range(len(l_cityCoord)):
        if i == index:
            eucDist.append(0)
            continue
        dist = round(sqrt((l_cityCoord[index][0] - l_cityCoord[i][0]) ** 2 + (l_cityCoord[index][1] - l_cityCoord[i][1]) ** 2), 2)
        eucDist.append(dist)

    return eucDist


def nearest_neighbour(cities):
    num_of_cities = len(cities)
    visited = [cities[0]]
    actual_city_index = cities.index(cities[0])

    route = []
    while True:
        while True:
            if len(visited) == num_of_cities:
                city_distance = euclidean_distance([visited[0], cities[actual_city_index]], 0)
                min_distance = city_distance[1]
                neighbour_city_index = 0
                break
            city_distance = euclidean_distance(cities, actual_city_index)
            min_distance = min([x for x in city_distance if x != 0 and x not in route])
            neighbour_city_index = city_distance.index(min_distance)
            if cities[neighbour_city_index] in visited:
                cities.remove(cities[neighbour_city_index])
                if neighbour_city_index < actual_city_index:
                    actual_city_index -= 1
                continue
            else:
                break
        route.append(min_distance)
        actual_city_index = neighbour_city_index
        visited.append(cities[actual_city_index])
        if len(route) == num_of_cities:
            break
    return round(sum(route), 2)

def dynamic_programming(mask, distances, starting_index):
    number_of_cities = len(distances[0])
    
    visited = (1 << number_of_cities) - 1
    
    if mask == visited:
        return distances[starting_index][0]

    cost = maxsize
    for city in range(number_of_cities):
        if mask & (1 << city) == 0:
            newCost = distances[starting_index][city] + dynamic_programming(mask | (1 << city), distances, city)
            cost = min(cost, newCost)
    return round(cost, 2)


if __name__ == "__main__":

    cities = city_coordinates(9)
    print(cities)

    distances = []
    for i in range(len(cities)):
        distances.append(euclidean_distance(cities, i))

    bruteForce = brute_force(distances, 0)
    print(bruteForce)

    nearestNeighbour = nearest_neighbour(cities)
    print(nearestNeighbour)

    dynamicPrograming = dynamic_programming(1, distances, 0)
    print(dynamicPrograming)