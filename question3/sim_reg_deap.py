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




INTERATIONS = 30

MAX_DEPTH       = 4
MAX_GENERATIONS = 200
POPULATION_SIZE = 100
MUTATION_RATE   = 0.25
CROSSOVER_RATE  = 0.90
TOURNAMENT_SIZE = 10
TERMINAL_SET    = np.linspace(-10,10,200)



def print_graph(X, hof, toolbox):
    colours = ['r','b','m','c','y']
    y = eval_all(X)

    pylab.plot(X,y,'g',linewidth=2.0)

    for c, ind in enumerate(hof):
        print str(ind)
        func = toolbox.compile(expr=ind)
        results   = np.array([])
        evaluated = 0.
        expected  = 0.

        z = eval_gp(X, func)
        pylab.plot(X, z, colours[c] ,linewidth=2.0)

    pylab.title("")
    pylab.subplots_adjust(top=0.8)

    pylab.show()



def get_pset_string(pset):
    function_set  = np.array([])
    for k,v in pset.primitives.items():
      for value in v:
          function_set = np.append(function_set, value.format('a','b'))

    return str( np.array2string(function_set, max_line_width=np.inf) )

def get_guess():

    results   = np.array([])
    evaluated = 0.
    expected  = 0.

    for (x) in np.linspace(-10,10,500):
        evaluated = guess(x)
        expected = regression_function(x)
        # print "evaluated: ", evaluated, " expected ", expected, " x ", x
        results = np.append(results, (evaluated-expected)**2 )

    return np.mean(results)

def get_mse(ind, toolbox):
    func = toolbox.compile(expr=ind)

    results   = np.array([])
    evaluated = 0.
    expected  = 0.

    for (x) in np.linspace(-10,10,500):
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
    header += "Terminal Set: np.linspace(-10,10,100)"

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

def eval_gues(X):
    results = np.array([])
    for x in X:
        evaluated = guess(x)
        results = np.append(results, evaluated)
    return results

def eval_all(x):
    results = np.array([])
    for val in x:
        expected = regression_function(val)
        results = np.append(results, expected)
    return results

def guess(a):
    fa = 0.0

    fa = safe_sqrt( safe_exp( (0.0 - 1.2) * a ) ) - 0.2

    return fa

def regression_function(a):
    fa = 0.0
    if (a > 0.):
        fa = (1/a) + math.sin(a)
    else:
        fa = a**2 + a*2 + 3.0
    return fa


def evaluate_individual(individual, points):
    score = 0.
    # get a callable function
    func = toolbox.compile(expr=individual)

    # find the squared errors

    results = np.array([])
    points = np.random.uniform(-10,10,200)
    for x in points:
        evaluated = func(x)
        expected = regression_function(x)
        results = np.append(results, (evaluated-expected)**2 )

    mae = np.mean(results)

    node_count = len(individual)

    score = mae
    score = ( node_count * 0.05 ) + mae

    return score,


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
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
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



print "Mse of guess: ", get_guess()
print ''


print "\nFunction Set: ", get_pset_string(pset)
print_header()
print ''

scores = np.array([])
scores_with_expression = np.array([])
temp_dict = {}

try:
    for i in range(INTERATIONS):
        random.seed(random.randint(0, sys.maxint))
        pop = toolbox.population(n=POPULATION_SIZE)
        hof = tools.HallOfFame(3)
        pop, log = algorithms.eaSimple(pop, toolbox, MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS,halloffame=hof, verbose=False)

        # print_summary(hof, toolbox)
        mse = get_mse(hof[0], toolbox)
        scores = np.append(scores, mse)

        temp_dict = {"score": mse, "best": hof[0]}
        scores_with_expression = np.append(scores_with_expression, temp_dict)
        print 'mse ', mse, ' exp: ', hof[0]

except KeyboardInterrupt:
    pass

# print('mean: ', np.mean(scores), ' std: ', np.std(scores))

print("%-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )

sorted_findings = sorted(scores_with_expression, key=lambda a: a['score'])

hof_hof = []

X = np.linspace(-10,10,500)
print "the three best expressions in order: "
for i in range(3):
    hof_hof.append(sorted_findings[i]['best'])
    print "mse: ", sorted_findings[i]['score'], "expression: ", sorted_findings[i]['best']
    print '----------'



X = np.linspace(-10,10,500)
print_graph(X, hof_hof, toolbox)




#
