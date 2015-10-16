from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import scipy.stats as stats
import numpy as np
import pandas as pd

def load_data(datafile, namefile):
    names = pd.read_csv(namefile, skiprows=1, header=None, sep=":").iloc[:,0].values
    names = np.append(names, 'klass')
    dt = pd.read_table(datafile, header=None, names=names, sep=",", index_col=False)
    X = dt.iloc[:,:-1]
    y = dt.iloc[:, -1]
    return X, y


def _pearsons_correlation(feature, y):
    return np.abs(stats.pearsonr(feature, y)[0])

def _rank(feature, y):
    # return _information_gain(feature, y)
    return _pearsons_correlation(feature, y)



def howgood(feature_names, x, y):
    gnb = GaussianNB()
    X_sub = x[feature_names].as_matrix()
    scores = cross_validation.cross_val_score( gnb, X_sub, y, cv=10)
    return np.mean(scores)

def best_chi2(X, y):
    best = [0.0, 0, []]
    column_names = X.columns.values

    for i in range(1, len(column_names)):
        ch2 = SelectKBest(chi2, k = i)
        ch2.fit(X, y)
        # X_train_features = ch2.fit_transform(X, y)

        selected = ch2.get_support()
        top_m    = [column_names[n] for (n, x) in enumerate(selected) if x == True]

        # print top_m
        # print bottom_m
        th = howgood(top_m, X, y)
        # print i, ' top m:  ', th
        if th > best[0]:
            best = th, i, top_m

    return best

def best_wrapper(X, y):

    best = [0.0, 0, []]
    column_names = X.columns.values

    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X, y)

    imp = np.array( clf.feature_importances_.tolist() )

    ranking_scores = {}
    for r in range(len(imp)):
        ranking_scores[column_names[r]] = imp[r]

    d = ranking_scores
    for i in range(1, len(column_names)):
        top_m    = [w for  w in sorted(d, key=d.get, reverse=True)][:i]

        th = howgood(top_m, X, y)
        # print i, ' top m:  ', th
        if th > best[0]:
            best = th, i, top_m

    return best



def best_pearsons(X, y):
    ranking_scores = {}
    column_names = X.columns.values

    for column_name in column_names:
        rank = _rank(X[column_name], y)
        # print "column name: ", column_name
        # howgood([column_name], x, y)
        ranking_scores[column_name] = rank

    d = ranking_scores
    # print "sorted ", [{w:d[w]} for  w in sorted(d, key=d.get, reverse=True)]

    best = [0.0, 0, []]
    for i in range(1, len(column_names)):
        top_m    = [w for  w in sorted(d, key=d.get, reverse=True)][:i]

        th = howgood(top_m, X, y)
        # print i, ' top m:  ', th
        if th > best[0]:
            best = th, i, top_m


    return best

def all(X, y):
    column_names = X.columns.values
    all_score = howgood(column_names, X, y)
    return all_score

#
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
#

names_file = 'data/sonar.names'
data_file = 'data/sonar.data'
X, y = load_data(data_file, names_file)

ch2_best = best_chi2(X, y)
print "best chi2:     ", ch2_best

pearsons_best = best_pearsons(X, y)
print "best pearsons: ", pearsons_best

wrapper_best = best_wrapper(X, y)
print "best wrapper:  ", wrapper_best

all_features = all(X, y)
print "all features:   ", all_features

#
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
#

names_file = 'data/wbcd.names'
data_file = 'data/wbcd.data'

X, y = load_data(data_file, names_file)

ch2_best = best_chi2(X, y)
print "best chi2:     ", ch2_best

pearsons_best = best_pearsons(X, y)
print "best pearsons: ", pearsons_best

wrapper_best = best_wrapper(X, y)
print "best wrapper:  ", wrapper_best

all_features = all(X, y)
print "all features:   ", all_features
