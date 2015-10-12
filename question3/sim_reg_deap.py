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



def print_graph(X, hof):
    colours = ['r','b','m','c','y']
    y = eval_all(X)

    pylab.plot(X,Y,'g',linewidth=2.0)

    for c, ind in enumerate(hof):

        func = toolbox.compile(expr=ind)

        results   = np.array([])
        evaluated = 0.
        expected  = 0.

        z = eval_gp(X, func)
        pylab.plot(X, z, colours[c] ,linewidth=1.0)

    pylab.title(summary)
    pylab.subplots_adjust(top=0.8)

    pylab.show()



def get_pset_string(pset):
    function_set  = np.array([])
    for k,v in pset.primitives.items():
      for value in v:
          function_set = np.append(function_set, value.format('a','b'))

    return str( np.array2string(function_set, max_line_width=np.inf) )

def get_mse(ind, toolbox):
    func = toolbox.compile(expr=ind)

    results   = np.array([])
    evaluated = 0.
    expected  = 0.

    for (x) in TERMINAL_SET:
        evaluated = func(x)
        expected = regression_function(x)
        # print "evaluated: ", evaluated, " expected ", expected, " x ", x
        results = np.append(results, (evaluated-expected)**2 )

    return np.mean(results)

def print_summary(hof, toolbox):

    for c, ind in enumerate(hof):
        mse = get_mse(ind, toolbox)
        print "mean squared error: ", mse, "  expression: ", ind




def print_header():
    header  = ""
    header += "Terminal Set: np.linspace(-20,20,100)"

    header += "\nPopulation Size: " + str(POPULATION_SIZE)
    header += "   Max Generations: " + str(MAX_GENERATIONS)
    header += "   Max Depth: " + str(MAX_DEPTH)

    header += "\nMutation Rate: " + str(MUTATION_RATE)
    header += "   Crossover Rate: " + str(CROSSOVER_RATE)
    header += "   Tournament Size: " + str(TOURNAMENT_SIZE)

    print header

def eval_gp(X, func):
    results = np.array([])
    for x in X:
        evaluated = func(x)
        results = np.append(results, evaluated)
    return results

def eval_all(x):
    results = np.array([])
    for val in x:
        expected = regression_function(val)
        results = np.append(results, expected)
    return results

def regression_function(a):
    fa = 0.0
    if (a > 0.):
        fa = (1/a) + math.sin(a)
    else:
        fa = a**2 + a*2 + 3.0
    return fa


def evaluate_individual(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    sqerrors = ((func(x) - regression_function(x))**2 for x in points)
    score = math.fsum(sqerrors) / len(points)

    return score,



# Define new functions
def safe_div(a,b):
    r = 0.
    if b != 0.:
        r = a / b

    return r

def safe_exp(a):
    safe_a = 50
    if abs(a) < 50:
        safe_a = a

    return math.exp( safe_a )

def safe_sqrt(a):
    return math.sqrt( abs(a) )


INTERATIONS = 30

MAX_DEPTH       = 4
MAX_GENERATIONS = 500
POPULATION_SIZE = 200
MUTATION_RATE   = 0.50
CROSSOVER_RATE  = 0.90
TOURNAMENT_SIZE = 5
TERMINAL_SET    = np.linspace(-20,20,200)

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.lt, 2)
pset.addPrimitive(operator.gt, 2)
pset.addPrimitive(operator.eq, 2)
pset.addPrimitive(operator.neg, 1)
# pset.addPrimitive(math.cos, 1)
# pset.addPrimitive(math.sin, 1)
pset.addPrimitive(safe_exp, 1)
pset.addPrimitive(safe_sqrt, 1)
pset.addEphemeralConstant("int10-10",lambda: random.randint(0, 20))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate_individual, points = TERMINAL_SET)
toolbox.register("select", tools.selTournament, tournsize = TOURNAMENT_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH))





print "\nFunction Set: ", get_pset_string(pset)
print_header()
print ''
#
#
# SEED = random.randint(0, sys.maxint)
# # SEED = 540263078815542890
# random.seed(random.randint(0, sys.maxint))
#
# pop = toolbox.population(n=POPULATION_SIZE)
# hof = tools.HallOfFame(3)
#
# pop, log = algorithms.eaSimple(pop, toolbox, MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS,
#                                halloffame=hof, verbose=False)
#
# print_summary(hof, toolbox)

scores = np.array([])

try:
    for i in range(INTERATIONS):
        random.seed(random.randint(0, sys.maxint))
        pop = toolbox.population(n=POPULATION_SIZE)
        hof = tools.HallOfFame(3)
        pop, log = algorithms.eaSimple(pop, toolbox, MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS,halloffame=hof, verbose=False)

        # print_summary(hof, toolbox)
        mse = get_mse(hof[0], toolbox)
        scores = np.append(scores, mse)
        print 'mse ', mse, ' exp: ', hof[0]

except KeyboardInterrupt:
    pass

# print('mean: ', np.mean(scores), ' std: ', np.std(scores))

print("%-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )



# write_graph(hof[0], "tree_logic.pdf")














X = np.linspace(-20,20,500)
#
# for x in X:
#     evaluated = func(x)
#     expected = regression_function(x)
#     # print "evaluated: ", evaluated, " expected ", expected, " x ", x
#     results = np.append(results, (evaluated-expected)**2 )
#
# mse = np.mean(results)
# print "expression: ", ind, "mean squared error: ", mse

# if c == 0:
#     header += "\nBest MSE: " + str(mse)
#     summary += "\n "



#
#
# nodes, edges, labels = gp.graph(ind)
#
# ### Graphviz Section ###
# import pygraphviz as pgv
#
# g = pgv.AGraph()
# g.add_nodes_from(nodes)
# g.add_edges_from(edges)
# g.layout(prog="dot")
#
# for i in nodes:
#     n = g.get_node(i)
#     n.attr["label"] = labels[i]
#
# g.draw("tree.pdf")
