from pyevolve import Util
from pyevolve import GTree
from pyevolve import GSimpleGA
from pyevolve import Consts
import math

rmse_accum = Util.ErrorAccumulator()

# def gp_add(a, b): return a+b
# def gp_sub(a, b): return a-b
# def gp_mul(a, b): return a*b
# def gp_sqrt(a):   return math.sqrt(abs(a))
def gp_or(a, b):  return a or b
def gp_and(a, b): return a and b
def gp_eq(a, b):  return a == b
def gp_not(a):    return not a


def eval_func(chromosome):
    score = 0.0

    code_comp = chromosome.getCompiledCode()

    values   = [ [1,1], [1,0], [0,0], [0,1]]
    expected = [ False, True,  False, True ]
    results  = []
    for i in values:
        a, b = i
        evaluated = eval(code_comp)
        results.append( evaluated )

    correct = len([x for x,val in enumerate(expected) if val == results[x]])
    height = chromosome.getHeight()
    score = ( height * 0.2 ) + ( len(values) - correct )

    return score

def main_run():
    genome = GTree.GTreeGP()
    genome.setParams(max_depth=4, method="ramped")
    genome.evaluator += eval_func

    ga = GSimpleGA.GSimpleGA(genome)
    ga.setParams(gp_terminals       = ['a', 'b'],
                 gp_function_prefix = "gp")

    ga.setMinimax(Consts.minimaxType["minimize"])
    # ga.setGenerations(100)
    ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    ga.setCrossoverRate(1.0)
    ga.setMutationRate(0.25)
    ga.setPopulationSize(100)

    ga(freq_stats=10)
    best = ga.bestIndividual()
    print best
    print "xor(a, b)"

    global rmse_accum
    rmse_accum.reset()
    code_comp = best.getCompiledCode()

    values   = [ [1,1], [1,0], [0,0], [0,1]]
    expected = [ False, True,  False, True ]


    for i, val in enumerate(values):
        a, b = val
        evaluated = eval(code_comp)
        target = expected[i]

        print("target: %-3.5f evaluated: %-3.5f" % (target, evaluated) ) ,
        print "\ta: ", a, " b: ", b, "  "







if __name__ == "__main__":
    main_run()
