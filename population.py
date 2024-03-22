import random

from individual import Individual
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Population:

    def __init__(self, X, target, mutation_rate):
        self.population = []
        self.generation = 0
        self.X = X
        self.target = target
        self.mutation_rate = mutation_rate
        self.best_individual = None
        self.finished = False
        self.perfect_score = 1.0
        self.max_fitness = 0.0
        self.average_fitness = 0.0
        self.mating_pool = []

    # Create a random initial population of individuals
    def create_initial_population(self, size):
        for i in range(size):
            ind = Individual(len(self.X[0]))
            ind.calculate_fitness(self.X, self.target)

            if ind.fitness > self.max_fitness:
                self.max_fitness = ind.fitness
                self.best_individual = ind

            self.average_fitness += ind.fitness
            self.population.append(ind)
        self.average_fitness /= size

    # Generate a mating pool based on the individual fitness (probability)
    def natural_selection(self):
        self.mating_pool = []

        for index, ind in enumerate(self.population):
            prob = int(round(ind.fitness * 100))
            self.mating_pool.extend([index for i in range(prob)])

    # Generate a new population from the mating pool
    def generate_new_population(self):
        new_population = []
        pop_size = len(self.population)
        self.average_fitness = 0.0

        for i in range(pop_size):
            partner_a, partner_b = self.selection()

            offspring = partner_a.crossover(partner_b)
            offspring.mutate(self.mutation_rate)
            offspring.calculate_fitness(self.X, self.target)

            self.average_fitness += offspring.fitness
            new_population.append(offspring)

        self.population = new_population
        self.generation += 1
        self.average_fitness /= pop_size

    def selection(self):
        pool_size = len(self.mating_pool)

        i_partner_a = random.randint(0, pool_size - 1)
        i_partner_b = random.randint(0, pool_size - 1)

        i_partner_a = self.mating_pool[i_partner_a]
        i_partner_b = self.mating_pool[i_partner_b]

        return self.population[i_partner_a], self.population[i_partner_b]

    # Evaluate the population
    def evaluate(self):
        best_fitness = 0.0

        for ind in self.population:
            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                self.max_fitness = best_fitness
                self.best_individual = ind

        if best_fitness == self.perfect_score:
            self.finished = True

    def print_population_status(self):
        print("\nGeneration: " + str(self.generation))
        print("Average fitness: " )
