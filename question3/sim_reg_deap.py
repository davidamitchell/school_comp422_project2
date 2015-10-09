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

# SEED = random.randint(0, sys.maxint)
SEED = 540263078815542890

MAX_DEPTH       = 4
MAX_GENERATIONS = 500
POPULATION_SIZE = 200

MUTATION_RATE   = 0.50
CROSSOVER_RATE  = 0.90

TOURNAMENT_SIZE = 5


terminal_set    = np.linspace(-20,20,100)
function_set    = np.array([])
for k,v in pset.primitives.items():
  for value in v:
      function_set = np.append(function_set, value.format('a','b'))

summary = ""
summary += "Function Set: "
summary += str( np.array2string(function_set, max_line_width=np.inf) )

summary += "\nTerminal Set: np.linspace(-20,20,100)"

summary += "\nPopulation Size: " + str(POPULATION_SIZE)
summary += "   Max Generations: " + str(MAX_GENERATIONS)
summary += "   Max Depth: " + str(MAX_DEPTH)

summary += "\nMutation Rate: " + str(MUTATION_RATE)
summary += "   Crossover Rate: " + str(CROSSOVER_RATE)
summary += "   Tournament Size: " + str(TOURNAMENT_SIZE)
summary += "   Seed: " + str(SEED)
# function_set


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)



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
    return math.fsum(sqerrors) / len(points),

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

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS,
                               stats=mstats,
                               halloffame=hof, verbose=False)
# # print hof
# for ind in hof:
#     print ind

colours = ['r','b','m','c','y']
X = np.linspace(-20,20,500)
Y = eval_all(X)
pylab.plot(X,Y,'g',linewidth=2.0)
# pylab.tight_layout()

for c, ind in enumerate(hof):

    func = toolbox.compile(expr=ind)

    results   = np.array([])
    evaluated = 0.
    expected  = 0.

    for x in X:
        evaluated = func(x)
        expected = regression_function(x)
        # print "evaluated: ", evaluated, " expected ", expected, " x ", x
        results = np.append(results, (evaluated-expected)**2 )

    # print results
    mse = np.mean(results)
    print "expression: ", ind, "mean squared error: ", mse

    if c == 0:
        summary += "\nBest MSE: " + str(mse)
        summary += "\n "

    # plot
    Z = eval_gp(X, func)
    pylab.plot(X,Z,colours[c] ,linewidth=2.0)


pylab.title(summary)
pylab.subplots_adjust(top=0.8)

pylab.show()

nodes, edges, labels = gp.graph(ind)

### Graphviz Section ###
import pygraphviz as pgv

g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]

g.draw("tree.pdf")


print summary

print ""



#
# add(add(mul(x, x), add(x, 3)), add(gt(x, 0), x)) * lt(eq(2, mul(13, x)), gt(lt(x, 10), add(x, x)))
#
#
#  mul(
#     add(add(mul(x, x), add(x, 3)), add(gt(x, 0), x)), lt(eq(2, mul(13, x)), gt(lt(x, 10), add(x, x)))
# )







# mul(add(gt(-7, x), safe_sqrt(x)), sub(lt(6, x), add(x, -6)))
#
#
#
#                      ( (-7 > x) + sqrt(x) ) * ( (6 < x) - (x - 6) )
