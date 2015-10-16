# gets 100% on data -- gp_sub(gp_mul(f4(), f3()), gp_mul(f2(), f1()))
#                      gp_sub ( gp_mul ( f4() f3()  )gp_mul ( f1() f2()  ) )

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
    X = dt.iloc[:,:-1].as_matrix()
    y = dt.iloc[:, -1].as_matrix()
    return X, y

def load_wine_data(filename):
    dt = pd.read_table(filename, header=None, names=["f1", "f2", "f3", "f4","f5", "f6", "f7", "f8","f9", "f10", "f11", "f12", "f13", "class"], sep=",", index_col=False)
    X = dt.iloc[:,:-1].as_matrix()
    y = dt.iloc[:, -1].as_matrix()
    return X, y

def gp_neg(a):      return gp_sub(0.,a)
def gp_add(a, b):   return a+b
def gp_sub(a, b):   return a-b
def gp_mul(a, b):   return a*b

def eval_func(chromosome):
    score = 0.0
    X = pd.DataFrame()

    code_comp = chromosome.getCompiledCode()
    X = eval(code_comp)
    X = X.reshape((len(X), 1))

    scores = cross_validation.cross_val_score( gnb, X, y_eval, cv=10)

    score = 1. -np.mean(scores)
    return score


def f1(): return X_eval[:,0]
def f2(): return X_eval[:,1]
def f3(): return X_eval[:,2]
def f4(): return X_eval[:,3]
def f5(): return X_eval[:,4]
def f6(): return X_eval[:,5]
def f7(): return X_eval[:,6]
def f8(): return X_eval[:,7]
def f9(): return X_eval[:,8]
def f10(): return X_eval[:,9]
def f11(): return X_eval[:,10]
def f12(): return X_eval[:,11]
def f13(): return X_eval[:,12]


def construct_feature():

    genome = GTree.GTreeGP()
    genome.setParams(max_depth=2, method="ramped",tournamentPool=5)
    genome.evaluator.set(eval_func)

    ga = GSimpleGA.GSimpleGA(genome)
    ga.setParams(gp_terminals       = ["f1()", "f2()", "f3()", "f4()"],
                 gp_function_prefix = "gp")

    ga.setMinimax(Consts.minimaxType["minimize"])
    ga.setGenerations(250)
    ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    ga.setCrossoverRate(0.9)
    ga.setMutationRate(0.50)
    ga.setPopulationSize(50)


    ga(freq_stats=0)
    best = ga.bestIndividual()
    return best


def construct_wine_feature():

    genome = GTree.GTreeGP()
    genome.setParams(max_depth=4, method="ramped",tournamentPool=5)
    genome.evaluator.set(eval_func)

    ga = GSimpleGA.GSimpleGA(genome)
    ga.setParams(gp_terminals       = ["f1()", "f2()", "f3()", "f4()","f5()", "f6()", "f7()", "f8()","f9()", "f10()", "f11()", "f12()", "f13()"],
                 gp_function_prefix = "gp")

    ga.setMinimax(Consts.minimaxType["minimize"])
    ga.setGenerations(250)
    ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
    ga.setCrossoverRate(0.9)
    ga.setMutationRate(0.50)
    ga.setPopulationSize(50)


    ga(freq_stats=0)
    best = ga.bestIndividual()
    return best





gnb = GaussianNB()
# X_all, y = load_data('data/balance.data')
#
# kf = cross_validation.StratifiedKFold(y, n_folds=10)
#
# scores = np.array([])
# full_scores = np.array([])
# print "balance: "
# try:
#     for k, (train, test) in enumerate(kf):
#
#         # allow updating of the X_eval used during
#         # GP training/feature selection
#         global X_eval
#         X_eval = X_all[train]
#         y_eval = y[train]
#
#         gp_programme = construct_feature()
#         print gp_programme.getSExpression()
#         code_comp = gp_programme.getCompiledCode()
#
#         X_train = eval(code_comp)
#         X_train = X_train.reshape((len(X_train), 1))
#
#         gnb.fit(X_train, y_eval)
#         train_score = gnb.score(X_train, y_eval)
#
#         X_eval = X_all[test]
#         y_eval = y[test]
#         X_test = eval(code_comp)
#         X_test = X_test.reshape((len(X_test), 1))
#
#         test_score = gnb.score(X_test, y_eval)
#         scores = np.append(scores, test_score)
#
#         print "training score      ", train_score, "      testing score ", test_score
#
#         gnb.fit(X_all[train], y[train])
#         full_train_score = gnb.score(X_all[train], y[train])
#         full_test_score = gnb.score(X_all[test], y[test])
#         full_scores = np.append(full_scores, full_test_score)
#         print "full training score ", full_train_score, " full testing score ", full_test_score
#
#
# except KeyboardInterrupt:
#     pass
#
# print("GP:   %-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )
#
# scores = cross_validation.cross_val_score( gnb, X_all, y, cv=10)
# print("ALL:  %-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )
# print("ALL:  %-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(full_scores), ' std: ', np.std(full_scores)) )


# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###


X_all, y = load_wine_data('data/wine.data')

kf = cross_validation.StratifiedKFold(y, n_folds=10)

scores = np.array([])
full_scores = np.array([])
print ""
print "="*80
print "wine: "
try:
    for k, (train, test) in enumerate(kf):

        # allow updating of the X_eval used during
        # GP training/feature selection
        global X_eval
        X_eval = X_all[train]
        y_eval = y[train]

        gp_programme = construct_wine_feature()
        print gp_programme.getSExpression()
        code_comp = gp_programme.getCompiledCode()

        X_train = eval(code_comp)
        X_train = X_train.reshape((len(X_train), 1))

        gnb.fit(X_train, y_eval)
        train_score = gnb.score(X_train, y_eval)

        X_eval = X_all[test]
        y_eval = y[test]
        X_test = eval(code_comp)
        X_test = X_test.reshape((len(X_test), 1))

        test_score = gnb.score(X_test, y_eval)
        scores = np.append(scores, test_score)

        print "training score      ", train_score, "      testing score ", test_score

        gnb.fit(X_all[train], y[train])
        full_train_score = gnb.score(X_all[train], y[train])
        full_test_score = gnb.score(X_all[test], y[test])
        full_scores = np.append(full_scores, full_test_score)
        print "full training score ", full_train_score, " full testing score ", full_test_score

except KeyboardInterrupt:
    pass


print("GP:   %-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )

scores = cross_validation.cross_val_score( gnb, X_all, y, cv=10)
print("ALL:  %-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )
print("ALL:  %-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(full_scores), ' std: ', np.std(full_scores)) )








#
