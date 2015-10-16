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

def best_chi2(X, y, n):

    best = [0.0, []]
    column_names = X.columns.values

    ch2 = SelectKBest(chi2, k = n)
    ch2.fit(X, y)

    selected = ch2.get_support()
    top_m    = [column_names[n] for (n, x) in enumerate(selected) if x == True]

    th = howgood(top_m, X, y)
    best = th, top_m

    return best

def best_wrapper(X, y, n):

    best = [0.0, []]
    column_names = X.columns.values

    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(X, y)

    imp = np.array( clf.feature_importances_.tolist() )

    ranking_scores = {}
    for r in range(len(imp)):
        ranking_scores[column_names[r]] = imp[r]

    d = ranking_scores
    top_m    = [w for  w in sorted(d, key=d.get, reverse=True)][:n]

    th = howgood(top_m, X, y)
    best = th, top_m

    return best



def best_pearsons(X, y, n):
    ranking_scores = {}
    column_names = X.columns.values

    for column_name in column_names:
        rank = _rank(X[column_name], y)
        # print "column name: ", column_name
        # howgood([column_name], x, y)
        ranking_scores[column_name] = rank

    d = ranking_scores
    # print "sorted ", [{w:d[w]} for  w in sorted(d, key=d.get, reverse=True)]

    best = [0.0, []]
    top_m    = [w for  w in sorted(d, key=d.get, reverse=True)][:n]

    th = howgood(top_m, X, y)
    best = th, top_m


    return best

def all(X, y):
    column_names = X.columns.values
    all_score = howgood(column_names, X, y)
    return all_score


def wrapper_score(X, y, n):

    column_counts = {}
    kf = cross_validation.StratifiedKFold(y, n_folds=10)

    scores = np.array([])
    full_scores = np.array([])
    for k, (train, test) in enumerate(kf):

        test_score, columns = best_wrapper(X.ix[train], y[train], n)
        scores = np.append(scores, test_score)
        for c in columns:
            column_counts[c] = column_counts.get(c, 0) + 1

    mean = np.mean(scores)
    std = np.std(scores)

    return scores, mean, std, column_counts

def pearsons_score(X, y, n):

    column_counts = {}
    kf = cross_validation.StratifiedKFold(y, n_folds=10)

    scores = np.array([])
    full_scores = np.array([])
    for k, (train, test) in enumerate(kf):

        test_score, columns = best_pearsons(X.ix[train], y[train], n)
        scores = np.append(scores, test_score)
        for c in columns:
            column_counts[c] = column_counts.get(c, 0) + 1

    mean = np.mean(scores)
    std = np.std(scores)

    return scores, mean, std, column_counts

def chi2_score(X, y, n):

    column_counts = {}
    kf = cross_validation.StratifiedKFold(y, n_folds=10)

    scores = np.array([])
    full_scores = np.array([])
    for k, (train, test) in enumerate(kf):

        test_score, columns = best_chi2(X.ix[train], y[train], n)
        scores = np.append(scores, test_score)
        for c in columns:
            column_counts[c] = column_counts.get(c, 0) + 1

    mean = np.mean(scores)
    std = np.std(scores)

    return scores, mean, std, column_counts

#
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
#

names_file = 'data/sonar.names'
data_file = 'data/sonar.data'
X, y = load_data(data_file, names_file)

print ""
print "sonar"

scores, mean, std, column_counts = chi2_score(X, y, 4)
print column_counts
print("best chi2:   %-6s %2.5f %-5s %2.5f \n" % ('mean: ', mean, ' std: ', np.std(scores)) )

scores, mean, std, column_counts = pearsons_score(X, y, 4)
print column_counts
print("best pearsons:   %-6s %2.5f %-5s %2.5f \n" % ('mean: ', mean, ' std: ', np.std(scores)) )

scores, mean, std, column_counts = wrapper_score(X, y, 4)
print column_counts
print("best wrapper:   %-6s %2.5f %-5s %2.5f \n" % ('mean: ', mean, ' std: ', np.std(scores)) )

all_features = all(X, y)
print "all features:   ", all_features

#
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
# #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ###
#

names_file = 'data/wbcd.names'
data_file = 'data/wbcd.data'
X, y = load_data(data_file, names_file)

print ""
print "wbcd"

scores, mean, std, column_counts = chi2_score(X, y, 4)
print column_counts
print("best chi2:   %-6s %2.5f %-5s %2.5f \n" % ('mean: ', mean, ' std: ', np.std(scores)) )

scores, mean, std, column_counts = pearsons_score(X, y, 4)
print column_counts
print("best pearsons:   %-6s %2.5f %-5s %2.5f \n" % ('mean: ', mean, ' std: ', np.std(scores)) )

scores, mean, std, column_counts = wrapper_score(X, y, 4)
print column_counts
print("best wrapper:   %-6s %2.5f %-5s %2.5f \n" % ('mean: ', mean, ' std: ', np.std(scores)) )

all_features = all(X, y)
print "all features:   ", all_features
