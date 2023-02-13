"""
In this question, you have the task to implement
(1) the Evolutionary Programming (EP)algorithm and
(2) either the Differential Evolution (DE) or Evolution Strategy (ES) algorithms.
You also need to apply the implemented algorithms to searching for the minimum of the following two functions, where D
is the number of variables, i.e., x1, x2, ..., xD.
(1) Rosenbrock function: f1(x1, x2, ..., xD) = ∑_{i=1}^{D−1}(1−xi)^2 + 100(xi+1−xi^2)^2
(2) Griewank's function: f2(x1, x2, ..., xD) = 1 + 1/4000 ∑i=1D(xi^2) − ∏i=1D cos(xi/sqrt(i))

For D = 20, do the following:
- Implement any specific variation of the EP and DE/ES algorithms of your choice. Justify your choice of the EP and
    DE/ES algorithm variations implement
- Choose appropriate algorithm parameters and population size, in line with your algorithm implementations.
- Determine the fitness function, solution encoding, and stopping criterion in EP and DE/ES.
- Since EP and DE/ES are stochastic algorithms, for each function, f1(x) or f2(x), repeat the experiments 30 times, and
    report the mean and standard deviation of your results, i.e., f1(x) or f2(x) values.
- Analyze your results, and draw your conclusions.

Then for D = 50, solve the Rosenbrock’s function using the same algorithm settings (repeat 30 times). Report the mean
and standard deviation of your results. Compare the performance of EP and DE/ES on the Rosenbrock’s function when D = 20
and D = 50. Analyze your results, and draw you conclusions.
"""
import copy
import operator
import os
import random
from functools import reduce
from operator import add

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import floor, log10
from common import utils


# Rosenbrock function
def rosenbrock(x):
    return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)])


# Griewank's function
def griewanks(x):
    return 1 + 1 / 4000 * sum([x[i] ** 2 for i in range(len(x))]) - np.prod(
        [np.cos(x[i] / np.sqrt(i + 1)) for i in range(len(x))])


# create a class for the individual
class Evolutionary_Individual:
    def __init__(self, D, bounds=(-30, 30)):
        self.D = D
        self.x = [random.uniform(np.min(bounds), np.max(bounds)) for i in range(D)]
        self.r = np.array([3.0 for _ in range(D)])
        self.f = np.inf
        self.tau = 1 / np.sqrt(2 * np.sqrt(D))
        self.tau_prime = 1 / np.sqrt(2 * D)

    # mutation with cauchy distribution
    def mutate(self, mu, sigma, indpb):
        self.x += self.r * np.random.standard_cauchy(size=self.D)  # delta_i times eta_i
        self.r *= np.exp(self.tau_prime * np.random.normal(0, 1) +
                         self.tau * np.random.standard_normal(size=self.D))

        return self

    def selection(self, **kwargs):
        # tournament selection
        # remove the current individual from the pool
        # in theory a true tournament pool should be used, but because this is python,
        # using the built-in sort function is much faster than iterating through the pool
        # with a for loop
        # fuck python yes but fuck python is convenient
        opponents = random.sample(kwargs.get('tournament_pool'), int(kwargs.get('q')))
        opponents += [self]
        opponents.sort(key=operator.attrgetter('f'))
        # get index of the self in the opponents
        self_index = opponents.index(self)

        return len(opponents) - self_index


class Differential_Individual(Evolutionary_Individual):
    def mutate(self, F=0.2, mate1=None, mate2=None):
        if mate1 is None or mate2 is None:
            raise Exception("Must provide two other individuals for reproduction")

        offspring = copy.deepcopy(self)

        offspring.x = offspring.x + F * np.subtract(np.array(mate1.x), np.array(mate2.x))
        return offspring

    def crossover(self, other, cxpb=0.4):
        n = len(self.x)
        j = random.randint(0, n)
        child = copy.deepcopy(self)
        child.x = np.asarray([self.x[i] if r <= cxpb or i == j else other.x[i]
                              for i, r in enumerate(np.random.uniform(size=n, low=0, high=1))])
        return child

    def selection(self, **kwargs):
        if self.f < kwargs.get('other').f:
            return self
        else:
            return kwargs.get('other')


def round(x, sf):
    return np.round(x, sf - int(floor(log10(abs(x)))) - 1)


# define the fitness function
def fitness(individual, f, bounds=(-30, 30)):
    # return the fitness value of the individual or inf if any of the x_i is out of the range [-30, 30]
    if any([x < np.min(bounds) or x > np.max(bounds) for x in individual.x]):
        return np.inf
    return f(individual.x)


def tournament_selection(all_individuals, N, q):
    """
    - Conduct pairwise comparison over the union of population[i] and offspring[i] for i = 1, ..., N.
    - Each solution is compared to q opponents
    - Select N solutions with the highest number of wins for the next
        generation
    """

    temp_individuals = all_individuals.copy()
    x = lambda ind: ind.selection(tournament_pool=list(set(temp_individuals[:]) - {ind}), q=N * q)
    all_individuals.sort(key=x, reverse=True)

    return all_individuals[:N]


def select_elites(population, N):
    """
    Select the N best individuals from the population
    """
    population.sort(key=operator.attrgetter('f'))
    return population[:N]


def evolutionary_programming(n_gen, f, N=100, D=20, mutpb=0.5, indpb=0.5, bounds=(-30, 30)):
    # Evaluate the individuals with an invalid fitness
    N = int(D * 5)
    population = [Evolutionary_Individual(D, bounds) for p in range(N)]
    fitnesses = [fitness(ind, f) for ind in population]
    for ind, fit in zip(population, fitnesses):
        ind.f = fit
    convergence = np.array([np.inf for _ in range(n_gen)])
    # Begin the evolution
    for g in range(n_gen):
        elites = select_elites(population, int(N * 0.1))
        offspring = list(map(copy.deepcopy, population))

        for i, mutant in enumerate(offspring):

            if random.random() < mutpb:
                mutant = mutant.mutate(indpb, 1, 0.5)
                mutant.f = fitness(mutant, f)
                offspring[i] = mutant

        # The population is entirely replaced by the offspring

        population[:] = tournament_selection(elites + population + offspring, N, 0.25)

        convergence[g] = population[0].f

    # return the fitness value for the best individual in the population
    return population, convergence


def differential_evolution(n_gen, f, N=100, D=20, mutpb=0.5, indpb=0.5, bounds=(-30, 30)):
    # create a population
    population = [Differential_Individual(D, bounds) for i in range(N)]
    fitnesses = [fitness(ind, f) for ind in population]
    convergence = np.array([np.inf for _ in range(n_gen)])
    for ind, fit in zip(population, fitnesses):
        ind.f = fit
    # begin the evolution
    for g in range(n_gen):
        # select the next generation individuals

        offspring = list(map(copy.deepcopy, population))
        # mutate = mutated ind if mutated ind is better than parent
        for i, mutant in enumerate(offspring):
            mate1, mate2, mate3 = random.sample(population, 3)
            while not (mutant != mate1 != mate2 != mate3 != mutant):
                mate1, mate2, mate3 = random.sample(population, 3)
            if random.random() < mutpb:
                # select three random individuals from the population
                mutant = mutant.mutate(mutpb, mate1, mate2)
            crossover = mutant.crossover(mate3)
            crossover.f = fitness(crossover, f)
            offspring[i] = crossover if crossover.f < population[i].f else offspring[i]
        population[:] = offspring
        # if the fitness is better, replace the individual in the population
        # population[:] = selection(population + offspring, N)
        convergence[g] = population[0].f
    # return the fitness value for the best individual in the population
    return population, convergence


def evaluate_metrics(fitness_values):
    metrics = {
        "mean": np.round(np.mean(fitness_values), 3),
        "std": np.round(np.std(fitness_values), 3),
        "min": np.round(np.min(fitness_values), 3),
        "max": np.round(np.max(fitness_values), 3),
    }
    return metrics


def main():
    max_iter = 30
    n_gen = 250
    pop_size = 100
    algorithms = [rosenbrock, griewanks]
    strategies = [evolutionary_programming, differential_evolution]

    D = 20
    metrics = {}
    print("For D =", D)
    for f in algorithms:
        print("\tRunning algorithm: {}".format(f.__name__))
        for strategy in strategies:
            print("\t\tOn strategy: {}".format(strategy.__name__))
            convergence = np.array([0 for _ in range(n_gen)], dtype=float)
            fitness_values = []
            for i in range(1, max_iter + 1):
                print("\t\t\titeration: {}".format(i))
                pop, conv = strategy(f=f, n_gen=n_gen, N=pop_size, D=D)
                fitness_values.append(pop[0].f)
                convergence = list(map(add, conv, convergence))

            print("---------------------------------------------------")
            metrics[f.__name__ + "_" + strategy.__name__] = evaluate_metrics(fitness_values)
            # plt.plot([f for f in fitness_values])
            # add line to the plot
            # plot the convergence
            plt.plot(range(0, n_gen), np.mean([convergence], axis=0))
            plt.xlabel("Generation")
            plt.ylabel('fitness')

            plt.title(f"{strategy.__name__} {f.__name__}  (D={D})")
            utils.output_plot(question_name=os.path.basename(__file__),
                              plot_name=f"{strategy.__name__}_{f.__name__}_D{D}", plot=plt)
            plt.show()

    results = pd.DataFrame(metrics, index=list(metrics.values())[0].keys(), columns=metrics.keys())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f"===\tResults for D={D}\t===")
        print(results)

    D = 50
    metrics = {}
    print("For D =", D)
    print("\tRunning algorithm: {}".format(rosenbrock.__name__))
    for strategy in strategies:
        print("\t\tOn strategy: {}".format(strategy.__name__))
        convergence = np.array([0 for _ in range(n_gen)], dtype=float)
        fitness_values = []
        for i in range(1, max_iter + 1):
            print("\t\t\titeration: {}".format(i))
            pop, conv = strategy(f=rosenbrock, n_gen=n_gen, N=pop_size, D=D)
            fitness_values.append(pop[0].f)
            convergence += conv
        print("---------------------------------------------------")
        metrics[strategy.__name__] = evaluate_metrics(fitness_values)
        plt.plot(range(0, n_gen), np.mean([convergence], axis=0))
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f"{strategy.__name__} {rosenbrock.__name__}  (D={D})")
        utils.output_plot(question_name=os.path.basename(__file__),
                          plot_name=f"{strategy.__name__}_{rosenbrock.__name__}_D{D}", plot=plt)
        plt.show()
        # pop = evolutionary_programming(0.5, 100, f1)
    results = pd.DataFrame(metrics, index=list(metrics.values())[0].keys(), columns=metrics.keys())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f"===Results for D={D}===")
        print(results)


if __name__ == "__main__":
    main()
