import matplotlib.pyplot as plt
import numpy as np
import random


import time
import itertools


# 25 Punkte
""" cities = np.array([(0, 16), (6, 56), (59, 77), (67, 35), (5, 13), 
                   (25, 16), (56, 4), (57, 42), (50, 70), (44, 8), 
                   (24, 11), (48, 63), (17, 88), (2, 69), (79, 35), 
                   (36, 64), (94, 66), (80, 18), (88, 59), (11, 1), 
                   (63, 45), (65, 8), (45, 91), (19, 78), (98, 66), 
                   ]) """

# 15 Punkte
""" cities = np.array([(0, 16), (6, 56), (59, 77), (67, 35), (5, 13), 
                   (25, 16), (56, 4), (57, 42), (50, 70), (44, 8), 
                   (24, 11), (48, 63), (17, 88), (2, 69), (79, 35), 
                   ]) """
                   
# 10 Punkte
cities = np.array([(0, 16), (6, 56), (59, 77), (67, 35), (5, 13), 
                   (25, 16), (56, 4),
                   ])

num_cities = len(cities)


def generate_population(pop_size, path_gen_string):
    return [random.sample(path_gen_string, len(path_gen_string)) for _ in range(pop_size)]

def calculate_distance(city1, city2):
    return np.sqrt((city1[0]-city2[0])**2 + (city1[1]-city2[1])**2)

def total_distance(path_gen_string):
    return sum(calculate_distance(cities[path_gen_string[i-1]], cities[path_gen_string[i]]) for i in range(len(path_gen_string)))

def calculate_fitness(path_gen_string):
    return total_distance(path_gen_string)

def select_parents(population):
    parent1, parent2 = roulette_wheel_selection(population)
    #parent1, parent2 = tournament_selection(population)
    #parent1, parent2 = rank_selection(population)
    
    return parent1, parent2
    
def roulette_wheel_selection(population):
    fitness_values = [calculate_fitness(path_gen_string) for path_gen_string in population]

    weights = [1 / (fitness ** 2) for fitness in fitness_values]
    parent1, parent2 = random.choices(population, k=2, weights=weights)
    while parent1 == parent2:
            parent1, parent2 = random.choices(population, k=2, weights=weights)
    return parent1, parent2 

def tournament_selection(population, tournament_size=3):
    def select_one_parent():
        tournament = random.sample(population, tournament_size)
        fitness_values = [calculate_fitness(path_gen_string) for path_gen_string in tournament]
        best_index = fitness_values.index(min(fitness_values))  # Assuming lower fitness is better
        return tournament[best_index]
    
    parent1 = select_one_parent()
    parent2 = select_one_parent()
    
    while parent1 == parent2:
        parent2 = select_one_parent()
    
    return parent1, parent2

def rank_selection(population):
    fitness_values = [calculate_fitness(path_gen_string) for path_gen_string in population]
    sorted_population = [x for _, x in sorted(zip(fitness_values, population))]
    ranks = list(range(1, len(sorted_population) + 1))
    
    total_ranks = sum(ranks)
    selection_probs = [rank / total_ranks for rank in ranks]
    
    parent1, parent2 = random.choices(sorted_population, k=2, weights=selection_probs)
    while parent1 == parent2:
        parent1, parent2 = random.choices(sorted_population, k=2, weights=selection_probs)
    
    return parent1, parent2

def create_child(parent1_part, parent2):
    child = parent1_part + [num for num in parent2 if num not in parent1_part]
    return child

def crossover(parent1, parent2):
    #child1, child2 = one_point_crossover(parent1, parent2)
    #child1, child2 = two_point_crossover(parent1, parent2)
    child1, child2 = uniform_crossover(parent1, parent2)
    return child1, child2

def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + [city for city in parent2 if city not in parent1[:point]]
    child2 = parent2[:point] + [city for city in parent1 if city not in parent2[:point]]
    return child1, child2

def two_point_crossover(parent1, parent2):
    point1 = random.randint(1, len(parent1) - 3)
    point2 = random.randint(point1 + 1, len(parent1) - 2)
    child1 = parent1[:point1] + [city for city in parent2 if city not in parent1[:point1]] + parent1[point2:]
    child2 = parent2[:point1] + [city for city in parent1 if city not in parent2[:point1]] + parent2[point2:]
    return child1, child2

def uniform_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    # Ensure valid TSP solutions (no duplicates, all cities present)
    child1 = fix_tsp_solution(child1, parent1, parent2)
    child2 = fix_tsp_solution(child2, parent1, parent2)
    return child1, child2

def fix_tsp_solution(child, parent1, parent2):
    """ Ensures the child is a valid TSP solution with no duplicates and all cities present. """
    cities = list(set(parent1).union(set(parent2)))
    missing_cities = list(set(cities) - set(child))
    city_count = {city: 0 for city in cities}

    for city in child:
        city_count[city] += 1

    duplicate_indices = [i for i, city in enumerate(child) if city_count[city] > 1]

    for i in duplicate_indices:
        if missing_cities:
            city_count[child[i]] -= 1
            child[i] = missing_cities.pop()

    return child

def mutate(path_gen_string, mutation_rate):
    if random.random() < mutation_rate:
        pos1, pos2 = random.sample(range(len(path_gen_string)), 2)
        path_gen_string[pos1], path_gen_string[pos2] = path_gen_string[pos2], path_gen_string[pos1]
    return path_gen_string

def evolve_population(population, mutation_rate):
    new_population = []
    while len(new_population) < len(population):
        parent1, parent2 = select_parents(population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_population.extend([child1, child2])
    return new_population

def genetic_algorithm(pop_size, path_gen_string, generations, mutation_rate):
    plt.ion() # Turn on interactive mode
    # Generate List
    best_solution = []
    best_solutions = []
    best_fitness_value = 0
    
    # Generate Population
    population = generate_population(pop_size, path_gen_string)
    for i in range(generations):
        population = evolve_population(population, mutation_rate)

        # Print Best Solution in this generation
        current_best = max(population, key=calculate_fitness)
        # Plot cities
        plt.scatter(cities[:,0], cities[:,1])
        for m in range(-1, len(current_best)-1):
            plt.plot((cities[current_best[m],0], cities[current_best[m+1],0]), (cities[current_best[m],1], cities[current_best[m+1],1]), 'r-')
    
        current_fitness = calculate_fitness(current_best)
        plt.title(f'Generation {i+1}, Distance: {current_fitness}')
        plt.draw()
        plt.pause(0.1)
        plt.clf()
        best_solutions.append(current_fitness)
        if best_fitness_value > current_fitness or best_fitness_value == 0:
            best_fitness_value = current_fitness
            best_solution = current_best
    return best_solution, best_solutions

def print_best_solution(solution, time):
    plt.ioff() # Turn off interactive mode

    # Plot the best path
    plt.scatter(cities[:,0], cities[:,1])
    for i in range(-1, len(solution)-1):
        plt.plot((cities[solution[i],0], cities[solution[i+1],0]), (cities[solution[i],1], cities[solution[i+1],1]), 'r-')
    plt.title(f'Best path with distance: {round(calculate_fitness(solution), 5)}, Total time: {round(time, 5)} seconds')
    plt.show()

def print_impovment_way(best_solutions):
    
    # Plot der besten Fitnesswerte über die Generationen
    plt.plot(range(1, generations + 1), best_solutions, marker='o', linestyle='-')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.title('Best Fitness Score over Generations')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    pop_size = 100
    generations = 10
    mutation_rate = 0.5

    # all cities bekommen eine number Gen String
    path_gen_string = list(range(num_cities))
    start_time = time.time()
    solution, best_solutions = genetic_algorithm(pop_size, path_gen_string, generations, mutation_rate)

    end_time = time.time() # End timer

    print_best_solution(solution, end_time - start_time)

    print_impovment_way(best_solutions)
    plt.ioff() # Turn off interactive mode

    

