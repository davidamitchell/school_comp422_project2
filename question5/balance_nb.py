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


def gp_add(a, b): return a+b
def gp_sub(a, b): return a-b
def gp_mul(a, b): return a*b
# def gp_div(a, b): return safe_div(a, b)

def eval_func(chromosome):
    score = 0.0
    a = X_all['f1']
    b = X_all['f2']

    X = pd.DataFrame()
    X['f3'] = X_all['f3']
    X['f4'] = X_all['f4']

    code_comp = chromosome.getCompiledCode()
    X['f1f2'] = eval(code_comp)

    scores = cross_validation.cross_val_score( gnb, X, y, cv=10)
    score = 1. -np.mean(scores)
    return score


#
# genome = GTree.GTreeGP()
# print genome.getParam('tournamentPool')
# genome.setParams(max_depth=2, method="ramped",tournamentPool=10)
# genome.evaluator += eval_func
#
# ga = GSimpleGA.GSimpleGA(genome)
# ga.setParams(gp_terminals       = ['a', 'b'],
#              gp_function_prefix = "gp")
#
# ga.setMinimax(Consts.minimaxType["minimize"])
# ga.setGenerations(50)
# ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
# ga.setCrossoverRate(0.9)
# ga.setMutationRate(0.25)
# ga.setPopulationSize(10)
#
#


X_all, y = load_data('data/balance.data')
X = X_all.as_matrix()
kf = cross_validation.KFold(len(X), n_folds=10, shuffle=False, random_state=None)

gnb = GaussianNB()
scores = np.array([])
for k, (train, test) in enumerate(kf):

    gnb.fit(X[train], y[train])
    scores = np.append(scores, gnb.score(X[test], y[test]))

# print scores
print np.mean(scores)



# ga(freq_stats=10)
# best = ga.bestIndividual()
# print best
#
#
# X = pd.DataFrame()
# X['f3'] = X_all['f3']
# X['f4'] = X_all['f4']
#
# a = X_all['f1']
# b = X_all['f2']
#
# code_comp = best.getCompiledCode()
# X['f1f2'] = eval(code_comp)
#
# scores = cross_validation.cross_val_score( gnb, X, y, cv=10)
# scores_all = cross_validation.cross_val_score( gnb, X_all, y, cv=10)
# print 'GP: ', np.mean(scores), 'All features: ', np.mean(scores_all)

#   for each fold
#       split training/test
#       using the training data
#       find a constructed feature
#           using GP
#           eval function is: a kfold validation across the above training
#
#       build training / test set using feature
#       preform the classification using constructed feature
#       evaluate performance
#










# perhaps final validation
# X, y = load_data('data/balance.data')
# scores = cross_validation.cross_val_score( gnb, X, y, cv=10)
#
# print 'X1'
# print scores
# print np.mean(scores)

# for each class
#  get data for that class
#  split
# use GP to find the best combination of features
#
#


#
# gnb = GaussianNB()
# X, y = load_data('data/balance.data')
# scores = cross_validation.cross_val_score( gnb, X, y, cv=10)
#
# print 'X'
# print scores
# print np.mean(scores)
#
# X1 = pd.DataFrame()
# X1['f1+f2'] = X['f1'] + X['f2']
# X1['f3'] = X['f3']
# X1['f4'] = X['f4']
#
#
# X, y = load_data('data/balance.data')
# scores = cross_validation.cross_val_score( gnb, X1, y, cv=10)
#
# print 'X1'
# print scores
# print np.mean(scores)
