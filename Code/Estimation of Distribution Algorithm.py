import array
import os
import random
from operator import add

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from common import utils


def load_data():
    data_files = utils.get_dataset_path("knapsack-data")
    data_files = [data_files + "/" + file for file in os.listdir(data_files)]
    data = {}
    for file in data_files:
        file_name = file.split("/")[-1]
        data[file_name] = {}
        content = pd.read_csv(file, sep=" ", header=None)
        # number of items in the knapscack is the first element on the first line
        data[file_name]["n_items"] = content.iloc[0, 0]
        # capacity of the knapsack is the second element on the first line
        data[file_name]["capacity"] = content.iloc[0, 1]
        # items are the rest of the lines in format: value weight
        data[file_name]["items"] = pd.DataFrame(content.iloc[1:, :].values, columns=["value", "weight"])
        # set value as index
        # data[file_name]["items"].set_index("value", inplace=True)
    return data


class PBIL(object):
    def __init__(self, ndim, learning_rate, mut_prob, mut_shift, lambda_):
        self.prob_vector = [0.5] * ndim
        self.learning_rate = learning_rate
        self.mut_prob = mut_prob
        self.mut_shift = mut_shift
        self.lambda_ = lambda_

    def sample(self):
        return (random.random() < prob for prob in self.prob_vector)

    def generate(self, ind_init):
        return [ind_init(self.sample()) for _ in range(self.lambda_)]

    def update(self, population):
        # best = heighest value with lowest weight
        best = max(population, key=lambda ind: ind.fitness.values[0] - ind.fitness.values[1])
        for i, value in enumerate(best):
            # Update the probability vector
            self.prob_vector[i] *= 1.0 - self.learning_rate
            self.prob_vector[i] += value * self.learning_rate

            # Mutate the probability vector
            if random.random() < self.mut_prob:
                self.prob_vector[i] *= 1.0 - self.mut_shift
                self.prob_vector[i] += random.randint(0, 1) * self.mut_shift


def eval_knapsack(individual):
    penal = MAX_WEIGHT/10
    weight = 0.0
    value = 0.0
    for item in items.itertuples():
        # print(individual[item.Index])
        if individual[item.Index]:
            weight += item.weight
            value += item.value
    penalty = penal * max(0, weight - MAX_WEIGHT)
    return value, penalty,


creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", array.array, typecode='b', fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("evaluate", eval_knapsack)


def main():
    global items, MAX_ITEM, MAX_WEIGHT
    knapsack_data = load_data()
    NGEN = 100
    optimal_values = [1514, 295, 9767]
    # Initialize the PBIL EDA

    for i, (name, dataset) in enumerate(knapsack_data.items()):
        convergence = [0] * NGEN
        fits = []
        for seed in range(5):
            random.seed(seed)
            np.random.seed(seed)
            if seed == 0:
                print(name)

            MAX_ITEM = dataset["n_items"]
            MAX_WEIGHT = dataset["capacity"]
            items = dataset["items"]
            pbil = PBIL(ndim=dataset.get("n_items"), learning_rate=0.5, mut_prob=0.1,
                        mut_shift=0.05, lambda_=1000)
            toolbox.register("generate", pbil.generate, creator.Individual)
            toolbox.register("update", pbil.update)

            # Statistics computation
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", numpy.mean, axis=0)
            stats.register("std", numpy.std, axis=0)
            stats.register("min", numpy.min, axis=0)
            stats.register("max", numpy.max, axis=0)

            pop, logbook = algorithms.eaGenerateUpdate(toolbox, NGEN, stats=stats, verbose=False)
            # get first axis of logbook avg
            convergence = list(
                map(add, [abs(optimal_values[i] - fit[0]) for fit in logbook.select("max")], convergence))
            fits.append(pop[0].fitness.values[0])
        print("mean:", np.mean(fits))
        print("std:", np.std(fits))
        plt.plot(range(NGEN), np.mean([convergence], axis=0), label=name)

        plt.title(name)
        plt.xlabel("Generation")
        plt.ylabel("Value (Distance from Optimal Value)")
        utils.output_plot(plot_name=name, question_name="knapsack", plot=plt)
        plt.show()
        # print([r for r in logbook.select("min")])
        # print([r for r in logbook.select("std")])
        # print("Best individual is %s, %s" % (max(pop, key=lambda ind: ind.fitness.values[0]), max(pop, key=lambda ind: ind.fitness.values[0]).fitness.values))
        # print this best solution
        # print(name)
        # print(pop[0].fitness.values[0])
        # print(pop[0].fitness.values[1])
        # get the rows from the items dataframe that are in the knapsack if the value in the pop is 1
        # print(len(pop))
        # # print first row of items
        # print(items.iloc[0])
        # solution = items[items.index[pop[0]] == 1]
        # print(solution["value"].sum())
        # print(solution["weight"].sum())
        # print(MAX_WEIGHT)
        # print(MAX_ITEM)
        # print(items.loc[pop[0]])
        # solution = [items[i] if item else '' for i, item in enumerate(pop[0])]
        # break


if __name__ == "__main__":
    main()
