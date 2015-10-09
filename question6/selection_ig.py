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
from sklearn.cross_validation import KFold


def load_data():
    namefile = 'data/sonar.names'
    names = pd.read_csv(namefile, skiprows=1, header=None, sep=":").iloc[:,0].values
    names = np.append(names, 'klass')
    datafile = 'data/sonar.data'
    dt = pd.read_table(datafile, header=None, names=names, sep=",", index_col=False)
    X = dt.iloc[:,:-1]
    y = dt.iloc[:, -1]
    return X, y


gnb = GaussianNB()

X_all, y_all = load_data()

X = X_all.as_matrix()
y = y_all.as_matrix()

kf = KFold(len(X), n_folds=10, shuffle=False, random_state=None)

scores = np.array([])
for k, (train, test) in enumerate(kf):
    gnb.fit(X[train], y[train])
    scores = np.append(scores, gnb.score(X[test], y[test]))


print scores
print np.mean(scores)


gnb = GaussianNB()
X = X_all.as_matrix()
y = y_all.as_matrix()
gnb.fit(X, y)

# kf = KFold(len(X), n_folds=10, shuffle=False, random_state=None)
#
# scores = np.array([])
# for k, (train, test) in enumerate(kf):
#     print "xall len: ", len(X), " y len ", len(y), "train: ", train
#     print X[train]
#     print y[train]
#     gnb.fit(X[train], y[train])
#     scores = np.append(scores, gnb.score(X[test], y[test]))
#
# print "feature1:"
# print scores
# print np.mean(scores)
