from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation


import scipy.stats as stats
import numpy as np
import pandas as pd

def load_data():
    namefile = 'data/sonar.names'
    names = pd.read_csv(namefile, skiprows=1, header=None, sep=":").iloc[:,0].values
    names = np.append(names, 'klass')
    datafile = 'data/sonar.data'
    dt = pd.read_table(datafile, header=None, names=names, sep=",", index_col=False)
    X = dt.iloc[:,:-1]
    y = dt.iloc[:, -1]
    return X, y


def _entropy(klasses):
    counts = np.bincount(klasses)
    probs = counts[np.nonzero(counts)] / float(len(klasses))
    return - np.sum(probs * np.log(probs))

def _information_gain(feature, y):
    # print feature
    pdf = stats.norm.pdf(feature)

    cond_pdf = cross_validation.cross_val_score( gnb, feature, y, cv=10)
    gnb = GaussianNB()


    print feature
    print "pdf ", pdf, pdf.sum()
    feature_set_indices = np.nonzero(feature)[0]

    feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices]

    print "feature set indices: ", feature_set_indices, len(feature_set_indices)
    print "feature_not_set_indices: ", feature_not_set_indices
    print "entropy before: ", entropy_before

    entropy_x_set = _entropy(y[feature_set_indices])
    entropy_x_not_set = _entropy(y[feature_not_set_indices])


    print "entropy_x_set: ", entropy_x_set
    print "entropy_x_not_set: ", entropy_x_not_set
    #
    # return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
    #                          + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))

    return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set) )


x, y = load_data()
x= x[['feature48','feature58']]
#
#
#
#
# gnb = GaussianNB()
# X = x['feature58'].as_matrix()
# X = X.reshape((len(X), 1))
#
# kf = cross_validation.StratifiedKFold(y, n_folds=10)
#
# scores = np.array([])
# for k, (train, test) in enumerate(kf):
#     gnb.fit(X[train], y[train])
#     scores = np.append(scores, gnb.score(X[test], y[test]))
#
#
# print scores
# print 'feature58: (nothing)', np.mean(scores)
#
#
# gnb = GaussianNB()
# X = x['feature48'].as_matrix()
# X = X.reshape((len(X), 1))
#
# kf = cross_validation.StratifiedKFold(y, n_folds=10)
#
# scores = np.array([])
# for k, (train, test) in enumerate(kf):
#     gnb.fit(X[train], y[train])
#     scores = np.append(scores, gnb.score(X[test], y[test]))
#
#
# print scores
# print 'feature48: (something) ', np.mean(scores)
#
#
# gnb = GaussianNB()
# X = x[['feature48','feature49','feature50','feature45']].as_matrix()
#
# kf = cross_validation.StratifiedKFold(y, n_folds=10)
#
# scores = np.array([])
# for k, (train, test) in enumerate(kf):
#     gnb.fit(X[train], y[train])
#     scores = np.append(scores, gnb.score(X[test], y[test]))
#
#
# print scores
# print 'feature48,feature49,feature50,feature45 (something) ', np.mean(scores)
#
#
#

feature_size = x.shape[0]
feature_range = range(0, feature_size)


entropy_before = _entropy(y)
information_gain_scores = {}


for column_name, column in x.transpose().iterrows():
    ig = _information_gain(column, y)
    print "column name: ", column_name
    information_gain_scores[column_name] = ig


print information_gain_scores
# exit()
#
# for feature in x.T:
#     print feature
#     information_gain_scores.append(_information_gain(feature, y))
