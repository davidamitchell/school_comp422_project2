from pyevolve import *

import numpy as np
import math
import pylab


def eval_gp(x, code_comp):
    results = np.array([])
    for val in x:
        a = b = val
        evaluated = eval(code_comp)
        results = np.append(results, evaluated)
    return results

def eval_all(x):
    results = np.array([])
    for val in x:
        expected = regression_function(val)
        results = np.append(results, expected)
    return results


def safe_ln(a):
    r = 0.
    if a > 0.:
        r = math.log(a)

    return r

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


def greater(a, b):
    if a > b:
        return a
    else:
        return b




def gp_add(a, b): return a+b
def gp_sub(a, b): return a-b
# def gp_mul(a, b): return a*b
# def gp_div(a, b): return safe_div(a, b)
def gp_neg(a):    return 0 - a
# def gp_gt(a, b):  return a<b
# def gp_lt(a, b):  return a<b
# def gp_or(a, b):  return a or b
# def gp_and(a, b): return a and b
def gp_eq(a, b):  return a == b
# def gp_not(a):    return not a
# def gp_sin(a):    return math.sin(a)
def gp_sqrt(a):   return math.sqrt(abs(a))
def gp_exp(a):    return safe_exp(a)
# def gp_greater(a, b):    return greater(a, b)
# def gp_ln(a):     return safe_ln(a)


def regression_function(a):
    fx = 0.0
    if (a > 0.):
        fx = (1/a) + math.sin(a)
    else:
        fx = a**2 + a*2 + 3.0
    return fx


def eval_func(chromosome):
    score = 0.0

    code_comp = chromosome.getCompiledCode()
    # values    =  (-5 - 20) * np.random.random_sample((500)) + 20
    values    = np.random.uniform(-10, 10, size=100)
    expected  = 0.0
    results   = np.array([])

    for val in values:
        a = b = val
        evaluated = eval(code_comp)
        expected = regression_function(a)
        results = np.append(results, abs(evaluated-expected) )

    mse = np.mean(results)

    height = chromosome.getHeight()
    # score = ( height * 0.2 ) + ( mse )
    score = mse

    return score

def main_run():
    genome = GTree.GTreeGP()
    print genome.getParam('tournamentPool')
    genome.setParams(max_depth=4, method="ramped")
    genome.mutator.set(Mutators.GTreeGPMutatorSubtree)
    genome.crossover.set(Crossovers.GTreeGPCrossoverSinglePoint)
    genome.evaluator.set(eval_func)

    ga = GSimpleGA.GSimpleGA(genome)
    ga.setParams(gp_terminals       = ['a', 'b'],
                 gp_function_prefix = "gp")

    ga.setMinimax(Consts.minimaxType["minimize"])
    ga.setGenerations(500)
    ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    # ga.selector.set(Selectors.GRouletteWheel)
    # print "selector ", ga.selector
    ga.setGPMode(True)
    ga.setCrossoverRate(0.9)
    ga.setMutationRate(0.25)
    ga.setPopulationSize(100)

    ga(freq_stats=100)


    colours = ['r','b','m','c','y']
    x = np.linspace(-10,10,100)
    y = eval_all(x)
    pylab.plot(x,y,'g',linewidth=2.0)

    pop = ga.getPopulation()
    for c, ind in enumerate(pop[:5]):

        code_comp = ind.getCompiledCode()
        results   = np.array([])
        evaluated = 0.
        expected  = 0.

        for val in x:
            a = b = val
            evaluated = eval(code_comp)
            expected = regression_function(a)
            results = np.append(results, abs(evaluated-expected) )


        print "expression: ", ind.getPreOrderExpression(), " score: ", ind.getFitnessScore(), "average error: ", np.mean(results)

        # plot
        z = eval_gp(x, code_comp)
        pylab.plot(x,z,colours[c] ,linewidth=2.0)

    # pylab.show()






if __name__ == "__main__":
    main_run()



# gp_sub(a, b)
#

# sqr( a + (a+a) ) - x
