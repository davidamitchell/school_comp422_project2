from pyevolve import Util
from pyevolve import GTree
from pyevolve import GSimpleGA
from pyevolve import Consts
import numpy as np
import math

rmse_accum = Util.ErrorAccumulator()

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




def gp_add(a, b): return a+b
def gp_sub(a, b): return a-b
def gp_mul(a, b): return a*b
def gp_gt(a, b):  return a<b
def gp_lt(a, b):  return a<b
def gp_or(a, b):  return a or b
def gp_and(a, b): return a and b
def gp_eq(a, b):  return a == b
def gp_not(a):    return not a
def gp_sin(a):    return math.sin(a)
def gp_sqrt(a):   return math.sqrt(abs(a))
# def gp_ln(a):     return safe_ln(a)


def regression_function(a):
    x = a
    fx = 0.0
    if (x > 0.):
        fx = (1/x) + math.sin(x)
    else:
        fx = x**2 + x*2 + 3.0

    return fx



def eval_func(chromosome):
    score = 0.0

    code_comp = chromosome.getCompiledCode()
    values    = np.random.uniform(-20, 50, size=1000)
    expected  = 0.0
    results   = np.array([])

    for val in values:
        a = b = val
        evaluated = eval(code_comp)
        expected = regression_function(a)
        results = np.append(results, (evaluated-expected) )

    mse = np.mean(results**2)

    height = chromosome.getHeight()
    score = ( height * 0.2 ) + ( mse )
    # score = mse

    return score

def main_run():
    genome = GTree.GTreeGP()
    genome.setParams(max_depth=5, method="ramped")
    genome.evaluator += eval_func

    ga = GSimpleGA.GSimpleGA(genome)
    ga.setParams(gp_terminals       = ['a', 'b'],
                 gp_function_prefix = "gp")

    ga.setMinimax(Consts.minimaxType["minimize"])
    ga.setGenerations(200)
    ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    ga.setCrossoverRate(1.0)
    ga.setMutationRate(0.25)
    ga.setPopulationSize(200)

    ga(freq_stats=50)
    best = ga.bestIndividual()
    print best
    print "crazy regression function"

    global rmse_accum
    rmse_accum.reset()
    code_comp = best.getCompiledCode()

    values   = [ -5,-4,-3,-2,-1,0,1,2,3,4,5 ]

    results   = np.array([])

    for i, val in enumerate(values):
        a = b = val
        evaluated = eval(code_comp)
        expected = regression_function(a)
        results = np.append(results, (evaluated-expected) )

        print("expected: %-3.5f evaluated: %-3.5f" % (expected, evaluated) ) ,
        print "\ta: ", a, " b: ", b, "  "

    mse = np.mean(results**2)
    print mse







if __name__ == "__main__":
    main_run()



# gp_sub(a, b)
#

# sqr( a + (a+a) ) - x
