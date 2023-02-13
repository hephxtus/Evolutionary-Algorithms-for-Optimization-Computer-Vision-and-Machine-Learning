"""
In project 1 we have tried to use GP to evolve a single genetic program to solve the following symbolic regression
problem:
    f(x) = x^-1 + sin(x), x > 0
    f(x) = 2x + x^2 + 3.0, x <= 0
In project 1, we assume that there is no prior knowledge about the target model. In this project, the assumption is
changed. Instead of knowing nothing, we know that the target model is a piecewise function, with two sub-functions f1(x)
for x > 0 and f2(x) for x ≤ 0. In other words, we know that the target function is:
    f(x) = f1(x) if x > 0
    f(x) = f2(x) if x ≤ 0
This question is to develop a Cooperative Co-evolution GP (CCGP) to solve this symbolic regression problem. The CCGP
should contain two sub-populations, one for f1(x) and the other for f2(x).
You can use a GP library. You should

- Determine and describe the terminal set and the function set of each sub-population.
- Design the fitness function and the fitness evaluation method for each sub-population.
- Set the necessary parameters, such as sub-population size, maximum tree depth, termination criteria, crossover and
    mutation rates.
- Run the implemented CCGP for 5 times with different random seeds. Report the best genetic programs (their structure
and performance) of each of the 5 runs. Present your observations and discussions and draw your conclusions.
"""

import math
import operator
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

import numpy as np
import pygraphviz
from deap import gp

STOP = NUM_SPECIES = 2


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False

import numpy

from deap import algorithms

import random

from deap import base
from deap import creator
from deap import tools

IND_SIZE = 64
POP_SIZE = 1000


def protectedDiv(left, right):
    try:
        return left / right
    except (ZeroDivisionError, OverflowError):
        return np.inf


def f(x):
    if x > 0:
        return (1 / x) + np.sin(x)
    else:
        return (2 * x) + (x ** 2) + 3.0


def evalFitness(individual, points):
    # Transform the tree expression in a callable function
    # print(individual)
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function
    sqerrors = []
    for x in points:
        try:
            # print(func(x), f(x))
            fn = func(x)
            fi = f(x)
            # convert to real number
            if isinstance(fn, complex):
                fn = fn.real
            if isinstance(fi, complex):
                fi = fi.real
            sqerrors.append((fn - fi) ** 2)
        except Exception as e:
            print(e)
            print(f"Error in individual {individual} for point {x}")
            return np.inf,
    try:
        return math.fsum(sqerrors) / len(points),
    except OverflowError:
        return np.inf,
    # return (math.fsum(sqerrors) / len(points),)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

pset = gp.PrimitiveSetTyped("MAIN", [float], float, "x")


def protectedSquare(x):
    try:
        return np.power(x, 2)
    except OverflowError:
        return np.inf


# pset = gp.PrimitiveSetTyped("main", [float], float, "x")
# pset.addPrimitive(operator.add, [float, float], float)
# pset.addPrimitive(operator.sub, [float, float], float)
# pset.addPrimitive(operator.mul, [float, float], float)
# pset.addPrimitive(protectedDiv, [float, float], float)
# pset.addPrimitive(math.cos, 1, float)
# pset = gp.PrimitiveSet()
pset.addPrimitive(operator.add, [float] * 2, name="add", ret_type=float)
pset.addPrimitive(operator.sub, [float] * 2, name="sub", ret_type=float)
pset.addPrimitive(operator.mul, [float] * 2, name="mul", ret_type=float)
pset.addPrimitive(operator.neg, [float] * 1, name="neg", ret_type=float)
pset.addPrimitive(protectedDiv, [float] * 2, name="div", ret_type=float)
pset.addPrimitive(np.cos, [float] * 1, name="cos", ret_type=float)
pset.addPrimitive(np.sin, [float] * 1, name="sin", ret_type=float)
pset.addPrimitive(lambda x: protectedDiv(1, x), [float] * 1, name='inv', ret_type=float)
# pset.addPrimitive(, 2, name="pow")
pset.addPrimitive(lambda x: math.sqrt(abs(x)), [float] * 1, name="sqrt", ret_type=float)
pset.addPrimitive(lambda x: protectedSquare(abs(x)), [float] * 1, name="pow2", ret_type=float)
# pset.addPrimitive(lambda x, y: protectedPow(abs(x), y), [float] * 2, name="pow", ret_type=float)
pset.addEphemeralConstant("rand101", lambda: random.random() * 100, float)
# pset.addPrimitive(operator.add, 2, name="add")
# pset.addPrimitive(np.cos, 1, name="cos")
# pset.addPrimitive(lambda x: protectedDiv(1, x), 1, name='inv')
# pset.addPrimitive(, 2, name="pow")
# pset.addPrimitive(lambda x: math.sqrt(abs(x)), 1, name="sqrt")

# pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(x0='x')

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("population", tools.initRepeat, list, toolbox.individual, POP_SIZE)
# toolbox.register("target_set", initTargetSet)

tournament_size = 3

toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("mate", gp.cxOnePoint)
# toolbox.register("mutate", tools.mutFlipBit, indpb=1. / IND_SIZE)
toolbox.register("get_best", tools.selBest, k=1, fit_attr='fitness')
# toolbox.register("evaluate", evalFitness, points=np.linspace(-10, 10, 100))
toolbox.register("evalPos", evalFitness, points=np.linspace(1, 75, 100))
toolbox.register("evalNeg", evalFitness, points=np.linspace(-50, 0, 100))

toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

NUM_population = 4
TARGET_SIZE = 30

import networkx as nx
from common import utils


def render_graph(hofs, seed, toolbox, question_name=os.path.basename(__file__), verbose=False):
    G = pygraphviz.AGraph(strict=False, directed=True)
    G.node_attr['style'] = 'filled'
    G.add_node(0)
    G.get_node(0).attr["label"] = "if"
    G.get_node(0).attr["fillcolor"] = "#cccccc"

    # LEFT PARENT
    G.add_node(1)
    G.get_node(1).attr["label"] = "x > 0"
    G.get_node(1).attr["fillcolor"] = "white"
    G.add_edge(0, 1)

    # RIGHT PARENT
    G.add_node(2)
    G.get_node(2).attr["label"] = "x ≤ 0"
    G.get_node(2).attr["fillcolor"] = "white"
    G.add_edge(0, 2)

    utils.render(G, hofs=hofs, seed=seed, toolbox=toolbox, question_name=question_name, verbose=verbose)


def main(seed, verbose=True):
    random.seed(seed)
    np.random.seed(seed)

    logbook = tools.Logbook()
    # logbook.header = "gen", "population", "evals", "std", "min", "avg", "max"

    ngen = 150

    # Initialize population
    species = [toolbox.population() for _ in range(STOP)]
    hofs = [tools.HallOfFame(1) for _ in range(STOP)]
    evals = [toolbox.evalPos, toolbox.evalNeg]

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("min", numpy.min)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("max", numpy.max)

    logbook.header = ['gen', 'nevals'] + mstats.fields

    # Init with random a representative for each species
    # representatives = [random.choice(s) for s in species]
    # Evaluate population
    for i1, (pop1, hof1, evaluatePos) in enumerate(zip(species, hofs, evals)):
        # Evaluate the individuals with an invalid fitness
        ind1 = [ind2 for ind2 in pop1 if not ind2.fitness.valid]
        fitnesses1 = toolbox.map(evaluatePos, ind1)
        for ind2, fit1 in zip(ind1, fitnesses1):
            ind2.fitness.values = fit1

        # Update the hall of fame with the generated individuals
        hof1.update(pop1)

        # next_repr = [None] * len(species)
    for i, (pop, hof, evaluate) in enumerate(zip(species, hofs, evals)):
        # Vary the species individuals
        g = 0
        while g < ngen:
            offspring = algorithms.varOr(pop, toolbox, lambda_=POP_SIZE, cxpb=0.8, mutpb=0.2)

            # Get the representatives excluding the current species
            # r = representatives[:i] + representatives[i + 1:]
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof.update(offspring)
            # Select the next representative
            # Select the individuals
            elite_size = 0.1
            pop[:] = tools.selBest(pop, int(len(pop) * elite_size)) + tools.selBest(offspring, int(len(
                pop) * (1 - elite_size)))  # Tournament selection
            # next_repr[i] = toolbox.get_best(offspring)[0]  # Best selection
            record = mstats.compile(offspring)
            logbook.record(gen=g, population=i, evals=len(invalid_ind), **record)

            if verbose:
                print(logbook.stream)

            g += 1
        # representatives = next_repr
    for hof in hofs:
        print("best individual for seed {} is: {} with fitness: {} and size: {}".format(seed, hof[0],
                                                                                        hof[0].fitness.values,
                                                                                        len(hof[0])))

    render_graph(hofs, seed, verbose=verbose, toolbox=toolbox, question_name=os.path.basename(__file__))


if __name__ == "__main__":
    for seed in range(1, 6):
        main(seed, verbose=False)

# describe the terminal set and the function set of each sub-population.
# The function set of the first sub-population is the same as the function set of the whole population.
# The function set of the second sub-population is the same as the function set of the first sub-population.

