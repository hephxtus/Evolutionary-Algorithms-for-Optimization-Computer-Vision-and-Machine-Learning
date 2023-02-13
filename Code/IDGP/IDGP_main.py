# python packages
import operator
import random

import IDGP.evalGP_main as evalGP
import IDGP.feature_function as fe_fs
# only for strongly typed GP
import IDGP.gp_restrict as gp_restrict
import numpy as np
from IDGP.strongGPDataType import Int1, Int2, Int3, Img, Region, Vector
# deap package
from deap import base, creator, tools, gp
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

# parameters:
population = 100
generation = 0
cxProb = 0.8
mutProb = 0.19
elitismProb = 0.01
totalRuns = 1
initialMinDepth = 2
initialMaxDepth = 6
maxDepth = 8


def setParameters(**kwargs):
    global population, generation, cxProb, mutProb, elitismProb, totalRuns, initialMinDepth, initialMaxDepth, maxDepth
    if 'population' in kwargs:
        population = kwargs['population']
    if 'generation' in kwargs:
        generation = kwargs['generation']
    if 'cxProb' in kwargs:
        cxProb = kwargs['cxProb']
    if 'mutProb' in kwargs:
        mutProb = kwargs['mutProb']
    if 'elitismProb' in kwargs:
        elitismProb = kwargs['elitismProb']
    if 'totalRuns' in kwargs:
        totalRuns = kwargs['totalRuns']
    if 'initialMinDepth' in kwargs:
        initialMinDepth = kwargs['initialMinDepth']
    if 'initialMaxDepth' in kwargs:
        initialMaxDepth = kwargs['initialMaxDepth']
    if 'maxDepth' in kwargs:
        maxDepth = kwargs['maxDepth']


##GP
def buildGP(data, x):
    bound1, bound2 = x[0, :, :].shape
    primitives = gp.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
    # Feature concatenation
    primitives.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector, name='FeaCon2')
    primitives.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector, name='FeaCon3')
    # Global feature extraction
    primitives.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
    primitives.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
    primitives.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
    primitives.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
    primitives.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')
    # Local feature extraction
    primitives.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
    primitives.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
    primitives.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
    primitives.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
    primitives.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')
    # Region detection operators
    primitives.addPrimitive(fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
    primitives.addPrimitive(fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
    # Terminals
    primitives.renameArguments(ARG0='Grey')
    #random string
    random_string = ''.join(random.choice('0123456789ABCDEF') for i in range(8))
    # check if the terminal is already added
    primitives = addEphemerals(primitives, random_string, bound1, bound2)
    # primitives.addEphemeralConstant('X'+random_string, lambda: random.randint(0, bound1 - 20), Int1)
    # primitives.addEphemeralConstant('Y'+random_string, lambda: random.randint(0, bound2 - 20), Int2)
    # primitives.addEphemeralConstant('Size'+random_string, lambda: random.randint(20, 51), Int3)
    return primitives


def addEphemerals(primitives, random_string, bound1, bound2):
    try:
        primitives.addEphemeralConstant('X'+random_string, lambda: random.randint(0, bound1 - 20), Int1)
        primitives.addEphemeralConstant('Y'+random_string, lambda: random.randint(0, bound2 - 20), Int2)
        primitives.addEphemeralConstant('Size'+random_string, lambda: random.randint(20, 51), Int3)
    except:
        r = ''.join(random.choice('0123456789ABCDEF') for i in range(8))
        primitives = addEphemerals(primitives, r, bound1, bound2)
    return primitives


# fitnesse evaluaiton
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def build_toolbox(df, x, y, **kwargs):
    primitives = buildGP(df, x)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=primitives, min_=initialMinDepth, max_=initialMaxDepth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitives)
    toolbox.register("mapp", map)

    # genetic operator
    toolbox.register("classifier", LinearSVC, max_iter=100)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("selectElitism", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitives)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
    toolbox.register("transform", transform, toolbox=toolbox)
    toolbox.register("evaluate", evaluate, toolbox=toolbox, x=x, y=y)
    return toolbox


def evaluate(individual, toolbox, x, y, cv=3, y_test=None, x_test=None):
    # print(individual)
    norm = toolbox.transform(individual, x=x, y=y)
    # print(train_norm.shape)
    lsvm = toolbox.classifier()
    if cv:
        metric = round(cross_val_score(lsvm, norm, y, scoring="f1", cv=cv).mean(), 6),
    else:
        norm_test = toolbox.transform(individual, x=x_test, y=y_test)
        lsvm.fit(norm, y)
        pred = lsvm.predict(norm_test)
        metric = (f1_score(pred, y_test), lsvm.score(norm_test, y_test))
    return metric
    # accuracy = round(100 * cross_val_score(toolbox.classifier(), norm, y, cv=3).mean(), 2)
    # return accuracy,


def evalTrainb(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        for i in range(0, len(y)):
            train_tf.append(np.asarray(func(x[i, :, :])))
        min_max_scaler = preprocessing.MinMaxScaler()
        train_norm = min_max_scaler.fit_transform(np.asarray(train_tf))
        lsvm = LinearSVC(max_iter=100)
        accuracy = round(100 * cross_val_score(lsvm, train_norm, y, cv=3).mean(), 2)
    except:
        accuracy = 0
    return accuracy,


def GPMain(randomSeeds, toolbox):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size_tree = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size_tree=stats_size_tree)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    log.header = ["gen", "evals"] + mstats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=mstats, halloffame=hof, verbose=True)

    return pop, log, hof


def transform(individual, toolbox, x, y):
    func = toolbox.compile(expr=individual)

    data = []
    for i in range(0, len(y)):
        # append data to 2dp
        data.append(np.asarray(func(x[i, :, :])))
    data = np.asarray(data)
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_data = min_max_scaler.fit_transform(np.asarray(data))

    return norm_data
