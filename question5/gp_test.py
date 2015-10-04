import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from pyevolve import Util
from pyevolve import GTree
from pyevolve import GSimpleGA
from pyevolve import Consts
import numpy as np
import math
import pylab


def load_data(filename):
    dt = pd.read_table(filename, header=None, names=["f1", "f2", "f3", "f4", "class"], sep=",", index_col=False)
    X = dt.iloc[:,:-1]
    y = dt.iloc[:, -1]
    return X, y


def safe_div(a,b):

    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.true_divide(a,b)
        r[r == np.inf] = 0
        r = np.nan_to_num(r)

    return r

def cond(a, b, c):
    if (a):
        return b
    else:
        return c


# need to think about how to put a
# conditional in here?

def gp_neg(a):      return gp_sub(0.,a)
def gp_add(a, b):   return a+b
def gp_sub(a, b):   return a-b
def gp_mul(a, b):   return a*b
def gp_div(a, b):   return safe_div(a, b)


X_all, y = load_data('data/balance.data')

def f1():        return X_all['f1']
def f2():        return X_all['f2']
def f3():        return X_all['f3']
def f4():        return X_all['f4']

# During the evaluation I want the terminal nodes to be
# features, or constants

def eval_func(chromosome):
    score = 0.0
    a = 2
    b = 4

    code_comp = chromosome.getCompiledCode()
    score = eval(code_comp)

    return np.mean(score**2)

genome = GTree.GTreeGP()
genome.setParams(max_depth=2, method="ramped",tournamentPool=10)
genome.evaluator.set(eval_func)

ga = GSimpleGA.GSimpleGA(genome)
ga.setParams(gp_terminals       = ["f1()", "f2()", "f3()", "f4()"],

# ga.setParams(gp_terminals       = ['a', 'b'],
             gp_function_prefix = "gp")

ga.setMinimax(Consts.minimaxType["minimize"])
ga.setGenerations(50)
ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
ga.setCrossoverRate(0.9)
ga.setMutationRate(0.25)
ga.setPopulationSize(10)

ga(freq_stats=10)
best = ga.bestIndividual()
print best
