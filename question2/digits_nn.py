import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sknn.mlp import Classifier, Convolution, Layer
from sklearn.neighbors import KNeighborsClassifier


import sys
import logging

logging.basicConfig(
            format="%(message)s",
            level=logging.ERROR,
            stream=sys.stdout)




def load_file(filename):
    return 0

def load_training(filename):
    dt = pd.read_table(filename, header=None, sep=" ")
    X = dt.iloc[:,:-1].values[500:].astype(float)
    y = dt.iloc[:, -1].values[500:]
    return X, y

def load_testing(filename):
    dt = pd.read_table(filename, header=None, sep=" ")
    X = dt.iloc[:,:-1].values[:500].astype(float)
    y = dt.iloc[:, -1:].values[:500]
    return X, y

def convolution(filename):

    X_train, y_train = load_training(filename)
    X_test, y_test = load_testing(filename)

    nn = Classifier(
        layers=[
            Convolution("Rectifier", channels=20, kernel_shape=(3,3)),
            Layer("Tanh", units=20),
            Layer("Softmax")],
        learning_rate=0.002,
        n_iter=50,
        verbose=False,
        debug=False)

    nn.fit(X_train, y_train)

    print "convolution score: ", nn.score(X_test, y_test)

    # print nn.score(X_test, y_test)
    # print confusion_matrix(y_test, nn.predict(X_test))


def standard(filename):

    X_train, y_train = load_training(filename)
    X_test, y_test = load_testing(filename)

    nn = Classifier(
        layers=[
            Layer("Tanh", units=10),
            Layer("Tanh", units=10),
            Layer("Softmax")],
        learning_rate=0.03,
        n_iter=200)

    nn.fit(X_train, y_train)

    print "standard 10x10 tanh score: ", nn.score(X_test, y_test)
    # print confusion_matrix(y_test, nn.predict(X_test))


def neighbor(filename):

    X_train, y_train = load_training(filename)
    X_test, y_test = load_testing(filename)

    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, y_train)

    print "knn score: ", clf.score(X_test, y_test)
    # print clf.score(X_test, y_test)
    # print confusion_matrix(y_test, clf.predict(X_test))

def run(filename):
    print ''
    print filename
    print "="*80
    neighbor(filename)
    standard(filename)
    convolution(filename)


run('data/digits00')
run('data/digits05')
run('data/digits40')
run('data/digits60')




# --------------
