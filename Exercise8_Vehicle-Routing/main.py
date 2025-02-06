import numpy as np
import random

# Define problem-specific constants
NUM_LOCATIONS = 10
NUM_VEHICLES = 3
MAX_CAPACITY = 50
MAX_ROUTE_TIME = 8 * 60  # in minutes
HIGH_PRIORITY_TIME_LIMIT = 120  # in minutes

# Coefficients for the fitness function
W1 = 10  # Weight for delay penalties
W2 = 5   # Weight for overcapacity penalties

# Generate random locations (x, y coordinates)
locations = np.random.rand(NUM_LOCATIONS, 2) * 100

# Generate dynamic distance matrices and traffic coefficients
def generate_dynamic_matrices():
    distances = np.linalg.norm(locations[:, np.newaxis] - locations[np.newaxis, :], axis=2)
    traffic = np.random.uniform(1.0, 2.0, (NUM_LOCATIONS, NUM_LOCATIONS))
    travel_times = distances * traffic
    return distances, traffic, travel_times

distances, traffic, travel_times = generate_dynamic_matrices()

# Priority and package weights
priorities = np.random.choice([0, 1], size=NUM_LOCATIONS, p=[0.8, 0.2])
weights = np.random.randint(5, 20, size=NUM_LOCATIONS)

# Central depot location (start and end point for all vehicles)
central_location = np.array([50, 50])

# Chromosome representation: list of routes, priorities, and capacities
def initialize_chromosome():
    routes = [list(np.random.permutation(NUM_LOCATIONS)) for _ in range(NUM_VEHICLES)]
    priorities_assignment = [[priorities[loc] for loc in route] for route in routes]
    capacities = [sum(weights[loc] for loc in route) for route in routes]
    return routes, priorities_assignment, capacities

# Debug print for chromosome details
def print_chromosome(chromosome, label="Chromosome"):
    routes, priorities_assignment, capacities = chromosome
    print(f"{label}:")
    for idx, (route, priority, capacity) in enumerate(zip(routes, priorities_assignment, capacities)):
        print(f"  Vehicle {idx+1}:")
        print(f"    Route: {route}")
        print(f"    Priorities: {priority}")
        print(f"    Capacity: {capacity}")
    print()

# Initialize population (chromosomes)
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        routes, priorities_assignment, capacities = initialize_chromosome()
        chromosome = (routes, priorities_assignment, capacities)
        population.append(chromosome)
        print_chromosome(chromosome, label="Initialized Chromosome")
    return population

# Fitness function
def fitness_function(chromosome):
    routes, _, capacities = chromosome
    total_distance = 0
    total_delay_penalty = 0
    total_overcapacity_penalty = 0

    for route in routes:
        capacity = 0
        route_time = 0
        for i, loc in enumerate(route):
            if i == 0:
                distance = np.linalg.norm(central_location - locations[loc])
            else:
                distance = distances[route[i-1], loc]

            travel_time = travel_times[route[i-1], loc] if i > 0 else 0

            total_distance += distance
            route_time += travel_time
            capacity += weights[loc]

            # High priority penalty
            if priorities[loc] and route_time > HIGH_PRIORITY_TIME_LIMIT:
                total_delay_penalty += 1

        # Back to central depot
        total_distance += np.linalg.norm(locations[route[-1]] - central_location)

        if capacity > MAX_CAPACITY:
            total_overcapacity_penalty += 1

        if route_time > MAX_ROUTE_TIME:
            total_delay_penalty += 1

    fitness = W1 * total_delay_penalty + W2 * total_overcapacity_penalty + total_distance
    return fitness

# Selection
def selection(population, fitnesses):
    selected = random.choices(population, weights=[1/f for f in fitnesses], k=2)
    return selected

# Crossover
def crossover(route1, route2):
    cut = random.randint(1, len(route1) - 1)
    child1 = route1[:cut] + [loc for loc in route2 if loc not in route1[:cut]]
    child2 = route2[:cut] + [loc for loc in route1 if loc not in route2[:cut]]
    return child1, child2

# Mutation
def mutate(route):
    if random.random() < 0.1:  # 10% mutation rate
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# Genetic Algorithm
def genetic_algorithm(pop_size, generations):
    population = initialize_population(pop_size)

    for generation in range(generations):
        fitnesses = [fitness_function(chromosome) for chromosome in population]
        new_population = []

        print(f"--- Generation {generation} ---")
        for idx, chromosome in enumerate(population):
            print_chromosome(chromosome, label=f"Chromosome {idx} (Fitness: {fitnesses[idx]})")

        for _ in range(pop_size // 2):
            parent1, parent2 = selection(population, fitnesses)
            child1_routes, _, _ = parent1
            child2_routes, _, _ = parent2

            new_child_routes1, new_child_routes2 = [], []
            for route1, route2 in zip(child1_routes, child2_routes):
                child1, child2 = crossover(route1, route2)
                new_child_routes1.append(mutate(child1))
                new_child_routes2.append(mutate(child2))

            new_population.append((
                new_child_routes1,
                [[priorities[loc] for loc in route] for route in new_child_routes1],
                [sum(weights[loc] for loc in route) for route in new_child_routes1]
            ))
            new_population.append((
                new_child_routes2,
                [[priorities[loc] for loc in route] for route in new_child_routes2],
                [sum(weights[loc] for loc in route) for route in new_child_routes2]
            ))

        population = new_population

        best_fitness = min(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    best_chromosome = population[np.argmin(fitnesses)]
    print_chromosome(best_chromosome, label="Best Chromosome (Solution)")
    return best_chromosome

# Run the genetic algorithm
best_solution = genetic_algorithm(pop_size=10, generations=5)
print("Best Solution:", best_solution)
