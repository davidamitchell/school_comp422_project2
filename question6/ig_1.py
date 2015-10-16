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

def _cond_entropy(feature, klasses):

    gnb = GaussianNB()

    x = feature.as_matrix()
    x = x.reshape((len(x), 1))

    y = klasses.as_matrix()


    probs_x   = stats.norm.pdf(feature)
    # print "probs_x ", probs_x


    gnb.fit(x, y)

    probs_c_x  = gnb.predict_proba(x)
    log_probs_c_x = np.log(probs_c_x)
    probs_c_x_sum = np.sum((log_probs_c_x * probs_c_x), axis=1)
    # print "probs_c_x ", probs_c_x

    # print "probs x: shape ", probs_x.shape, " probs c x shape ", probs_c_x.shape, " probs c x sum shape ", probs_c_x_sum.shape

    # print "log_probs_c_x ", log_probs_c_x

    ce = np.sum((log_probs_c_x * probs_c_x), axis=1) * probs_x

    cond_entropy = - np.sum(ce)
    # print "cond_entropy: ", cond_entropy
    return cond_entropy


def _information_gain(feature, y):

    pdf = stats.norm.pdf(feature)
    #
    # print feature
    # print "pdf ", pdf, pdf.sum()

    cond_entropy = _cond_entropy(feature, y)
    info_gain = entropy_before - ( cond_entropy )
    return info_gain


x, y = load_data()
#
#
#

def howgood(feature_names, x, y):
    gnb = GaussianNB()
    X = x[feature_names].as_matrix()
    # X = X.reshape((len(X), 1))

    kf = cross_validation.StratifiedKFold(y, n_folds=10)

    scores = np.array([])
    for k, (train, test) in enumerate(kf):
        gnb.fit(X[train], y[train])
        scores = np.append(scores, gnb.score(X[test], y[test]))

    # print feature_names, '\n score: ', np.mean(scores)
    print np.mean(scores)

#

feature_size = x.shape[0]
feature_range = range(0, feature_size)


entropy_before = _entropy(y)
information_gain_scores = {}


for column_name, column in x.transpose().iterrows():
    ig = _information_gain(column, y)
    # print "column name: ", column_name
    # howgood([column_name], x, y)
    information_gain_scores[column_name] = ig




# print information_gain_scores

d = information_gain_scores
# print "sorted ", [{w:d[w]} for  w in sorted(d, key=d.get, reverse=True)]

m = 35
for i in range(59):
    top_m    = [w for  w in sorted(d, key=d.get, reverse=True)][:i+1]
    bottom_m = [w for  w in sorted(d, key=d.get, reverse=False)][:i+1]

    # print top_m
    # print bottom_m
    print i, ' top m:  ',
    howgood(top_m, x, y)
    print i, ' bottom: ',
    howgood(bottom_m, x, y)
#

X = x.as_matrix()
gnb = GaussianNB()
kf = cross_validation.StratifiedKFold(y, n_folds=10)
scores = np.array([])
for k, (train, test) in enumerate(kf):
    gnb.fit(X[train], y[train])
    scores = np.append(scores, gnb.score(X[test], y[test]))

print 'all:    ', np.mean(scores)
