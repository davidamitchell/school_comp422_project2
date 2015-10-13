# gets 100% on data -- gp_sub(gp_mul(f4(), f3()), gp_mul(f2(), f1()))

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
    r = 0.
    if b != 0.:
        r = a / b

    return r

def cond(a, b, c):
    if (a):
        return b
    else:
        return c


# need to think about how to put a
# conditional in here?

def gp_neg(a):      return gp_sub(0.,a)
# def gp_if(a, b, c): return cond(a,b,c)
def gp_add(a, b):   return a+b
def gp_sub(a, b):   return a-b
def gp_mul(a, b):   return a*b
# def gp_div(a, b):   return safe_div(a, b)
# def gp_f1():        return X_eval['f1']
# def gp_f2():        return X_eval['f2']
# def gp_f3():        return X_eval['f3']
# def gp_f4():        return X_eval['f4']

# During the evaluation I want the terminal nodes to be
# features, or constants

def eval_func(chromosome):
    score = 0.0
    X = pd.DataFrame()

    code_comp = chromosome.getCompiledCode()
    X = eval(code_comp)
    X = X.reshape((len(X), 1))

    scores = cross_validation.cross_val_score( gnb, X, y, cv=10)

    score = 1. -np.mean(scores)
    # print score
    return score


def f1():        return X_eval['f1']
def f2():        return X_eval['f2']
def f3():        return X_eval['f3']
def f4():        return X_eval['f4']


gnb = GaussianNB()
X_all, y = load_data('data/balance.data')
X_eval = X_all

genome = GTree.GTreeGP()
genome.setParams(max_depth=2, method="ramped",tournamentPool=10)
genome.evaluator.set(eval_func)

ga = GSimpleGA.GSimpleGA(genome)
ga.setParams(gp_terminals       = ["f1()", "f2()", "f3()", "f4()"],
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

code_comp = best.getCompiledCode()
X = eval(code_comp)
X = X.reshape((len(X), 1))
score = cross_validation.cross_val_score( gnb, X, y, cv=10)
print score
