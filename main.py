from population import Population
from sklearn.datasets import load_breast_cancer


def genetic_algorithm():
    pop_size = 50
    mutation_rate = 0.01

    # Load the breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)
    target = y

    pop = Population(X, target, mutation_rate)
    pop.create_initial_population(pop_size)

    while not pop.finished:
        pop.natural_selection()
        pop.generate_new_population()
        pop.evaluate()
        pop.print_population_status()


if __name__ == '__main__':
    genetic_algorithm()
