# Vehicle Routing Problem with Genetic Algorithm - Report

This report details the implementation of a Genetic Algorithm (GA) to solve a Vehicle Routing Problem (VRP). The problem involves optimizing routes for multiple vehicles to serve a set of locations with varying priorities and package weights, while considering capacity constraints and time limits.

## 1. Problem Definition

The VRP aims to minimize the total cost, which includes travel distance, penalties for delays (especially for high-priority locations), and penalties for exceeding vehicle capacity.  The problem is defined by:

*   `NUM_LOCATIONS`: Number of locations to be served.
*   `NUM_VEHICLES`: Number of vehicles available.
*   `MAX_CAPACITY`: Maximum weight a vehicle can carry.
*   `MAX_ROUTE_TIME`: Maximum time a vehicle can spend on a route.
*   `HIGH_PRIORITY_TIME_LIMIT`: Time limit for serving high-priority locations.
*   `W1`: Weight for delay penalties in the fitness function.
*   `W2`: Weight for overcapacity penalties in the fitness function.

Locations are randomly generated with (x, y) coordinates.  Dynamic distance matrices and traffic coefficients are also generated, making the travel times between locations variable. Each location has an assigned priority (high or low) and a package weight. A central depot serves as the starting and ending point for all vehicle routes.

## 2. Solution Approach

A Genetic Algorithm is employed to find a near-optimal solution to the VRP. The key components of the GA are:

### 2.1. Chromosome Representation

A chromosome represents a solution to the VRP. It consists of:

*   `routes`: A list of routes, where each route is a list of location indices visited by a vehicle.
*   `priorities_assignment`: A list mirroring the routes, containing the priority of each location in the respective route.
*   `capacities`: A list containing the total weight carried by each vehicle for its assigned route.

### 2.2. Initialization

An initial population of chromosomes is generated randomly. Each chromosome is created by randomly assigning locations to vehicles and calculating the corresponding priorities and capacities.

### 2.3. Fitness Function

The fitness function evaluates the quality of a chromosome (solution). It calculates the total cost based on:

*   Total travel distance.
*   Delay penalties: Penalties are added if high-priority locations are served after `HIGH_PRIORITY_TIME_LIMIT` or if a route exceeds `MAX_ROUTE_TIME`.
*   Overcapacity penalties: Penalties are added if a vehicle's capacity is exceeded.

The fitness is calculated as: `fitness = W1 * total_delay_penalty + W2 * total_overcapacity_penalty + total_distance`.  Lower fitness values indicate better solutions.

### 2.4. Selection

The selection process chooses two parent chromosomes from the population based on their fitness.  Inverse fitness weighting is used, giving better solutions a higher probability of being selected.  `random.choices` with weights is used for this.

### 2.5. Crossover

The crossover operation combines the genetic material of two parent chromosomes to create new offspring. A single-point crossover is used: a random cut point is chosen, and the routes are swapped between the parents up to that point. Locations not already present in the child's route are then appended.

### 2.6. Mutation

The mutation operator introduces small random changes to the routes to explore new solutions.  A swap mutation is used, where two randomly chosen locations within a route are swapped.  A 10% mutation rate is applied.

### 2.7. Genetic Algorithm Loop

The GA iteratively performs the following steps:

1.  Initialize a population of chromosomes.
2.  Evaluate the fitness of each chromosome.
3.  Select parents based on fitness.
4.  Perform crossover and mutation to create offspring.
5.  Replace the old population with the new offspring.
6.  Repeat steps 2-5 for a fixed number of generations.

The best chromosome found during the evolution process is returned as the solution.

## 3. Results

The genetic algorithm is run with a population size of 10 and for 5 generations. The report includes printouts of the chromosome details (routes, priorities, capacities) and their fitness scores for each generation.  The best chromosome found after the final generation is printed as the "Best Solution".  Due to the random nature of the algorithm, the results will vary on each run.

## 4. Discussion

This implementation demonstrates a basic GA approach to solving the VRP. The fitness function incorporates the key constraints and objectives of the problem. The crossover and mutation operators are designed to explore the solution space effectively.

Further improvements could include:

*   Exploring different crossover and mutation operators.
*   Tuning the GA parameters (population size, mutation rate, etc.).
*   Implementing more sophisticated selection strategies.
*   Adding local search heuristics to further refine the solutions.
*   Using more advanced VRP formulations and algorithms.

This report provides a comprehensive overview of the implemented GA for the VRP.  The code provides a working foundation that can be extended and improved upon.