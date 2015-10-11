print ""

import operator
import math
import random
import sys
import numpy as np
import pylab

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import pygraphviz as pgv

def xor_func(a, b):
    return int( operator.xor( bool(a), bool(b) ) )

def evaluate_individual(individual, points):
    score = 0.
    # get a callable function
    func = toolbox.compile(expr=individual)

    # find the squared errors

    results = np.array([])
    for (a, b) in points:
        evaluated = func(a, b)
        expected = xor_func(a, b)
        results = np.append(results, abs(evaluated-expected) )

    mae = np.mean(results)

    node_count = len(individual)

    score = mae
    score = ( node_count * 0.05 ) + mae

    return score,

def write_graph(ind, filename):

    nodes, edges, labels = gp.graph(ind)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(filename)

def get_pset_string(pset):
    function_set  = np.array([])
    for k,v in pset.primitives.items():
      for value in v:
          function_set = np.append(function_set, value.format('a','b'))

    return str( np.array2string(function_set, max_line_width=np.inf) )


def print_summary(hof, toolbox, summary):

    for c, ind in enumerate(hof):

        func = toolbox.compile(expr=ind)

        results   = np.array([])
        evaluated = 0.
        expected  = 0.

        for (a, b) in terminal_set:
            evaluated = func(a, b)
            expected = xor_func(a, b)
            # print "evaluated: ", evaluated, " expected ", expected, " x ", x
            results = np.append(results, (evaluated-expected)**2 )

        # print results
        mse = np.mean(results)
        print "expression: ", ind, "mean squared error: ", mse

        if c == 0:
            summary += "\nBest MSE: " + str(mse)
            summary += "\n "


    print summary

    print ""



SEED = random.randint(0, sys.maxint)
# SEED = 540263078815542890

MAX_DEPTH       = 4
MAX_GENERATIONS = 100
POPULATION_SIZE = 50

MUTATION_RATE   = 0.25
CROSSOVER_RATE  = 0.90

TOURNAMENT_SIZE = 3

terminal_set    = np.array( [ [1,1], [1,0], [0,0], [0,1] ] )


# summary += "\nTerminal Set: ", str( terminal_set.tolist() )

summary = ""
summary += "\nPopulation Size: " + str(POPULATION_SIZE)
summary += "   Max Generations: " + str(MAX_GENERATIONS)
summary += "   Max Depth: " + str(MAX_DEPTH)

summary += "\nMutation Rate: " + str(MUTATION_RATE)
summary += "   Crossover Rate: " + str(CROSSOVER_RATE)
summary += "   Tournament Size: " + str(TOURNAMENT_SIZE)
summary += "   Seed: " + str(SEED)
# function_set

# ##########################################################################################################################################
# ##########################################################################################################################################
# ##########################################################################################################################################

pset_logic = gp.PrimitiveSet("Logic", 2)
pset_logic.addPrimitive(operator.or_, 2)
pset_logic.addPrimitive(operator.not_, 1)
pset_logic.addPrimitive(operator.and_, 2)
pset_logic.addPrimitive(operator.eq, 2)
pset_logic.addPrimitive(operator.ne, 2)
pset_logic.renameArguments(ARG0='a')
pset_logic.renameArguments(ARG1='b')

# ########################################

pset = pset_logic
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


toolbox.register("evaluate", evaluate_individual, points = terminal_set)
toolbox.register("select", tools.selTournament, tournsize = TOURNAMENT_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))

random.seed(SEED)

pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(3)

pop, log = algorithms.eaSimple(pop, toolbox, MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS,
                               halloffame=hof, verbose=False)


print "\nFunction Set: ", get_pset_string(pset)
print_summary(hof, toolbox, summary)
write_graph(hof[0], "tree_logic.pdf")

# ##########################################################################################################################################
# ##########################################################################################################################################
# ##########################################################################################################################################


pset_arth = gp.PrimitiveSet("Arthmatic", 2)
pset_arth.addPrimitive(operator.add, 2)
pset_arth.addPrimitive(operator.sub, 2)
pset_arth.addPrimitive(operator.mul, 2)
pset_arth.addPrimitive(operator.neg, 1)
pset_arth.renameArguments(ARG0='a')
pset_arth.renameArguments(ARG1='b')

# ########################################

pset = pset_arth
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


toolbox.register("evaluate", evaluate_individual, points = terminal_set)
toolbox.register("select", tools.selTournament, tournsize = TOURNAMENT_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))

random.seed(SEED)

pop = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(3)

pop, log = algorithms.eaSimple(pop, toolbox, MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS,
                               halloffame=hof, verbose=False)


print "\nFunction Set: ", get_pset_string(pset)
print_summary(hof, toolbox, summary)
write_graph(hof[0], "tree_arth.pdf")
