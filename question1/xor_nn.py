import sys
import numpy as np
import pandas as pd
from sknn.mlp import Classifier, Layer


import random

def load_data(filename):
    dt = pd.read_table(filename, header=None, sep=" ")
    X = dt.iloc[:,:-1].as_matrix()
    y = dt.iloc[:, -1].as_matrix()
    return X, y

def tanh(X, y, nodes):

    seed = random.randint(0, sys.maxint)

    X_train, y_train = X, y
    X_test, y_test = X, y

    nn = Classifier(
        layers=[
            Layer("Tanh", units=nodes),
            Layer("Softmax")],
        learning_rate=0.2,
        batch_size=1,
        n_iter=1000,
        random_state=seed,
        verbose=False,
        debug=False
        )

    nn.fit(X_train, y_train)

    train_score = nn.score(X_train, y_train)
    test_score = nn.score(X_test, y_test)
    proba = nn.predict_proba(X_test)

    return train_score, test_score, proba

def maxout(X, y, nodes):

    seed = random.randint(0, sys.maxint)

    X_train, y_train = X, y
    X_test, y_test = X, y

    nn = Classifier(
        layers=[
            Layer("Maxout", units=nodes, pieces=2),
            Layer("Softmax")],
        learning_rate=0.2,
        batch_size=1,
        n_iter=200,
        random_state=seed,
        verbose=False,
        debug=False
        )

    nn.fit(X_train, y_train)

    train_score = nn.score(X_train, y_train)
    test_score = nn.score(X_test, y_test)
    proba = nn.predict_proba(X_test)

    return train_score, test_score, proba




def run(interations, method, X, y, nodes):
    scores = np.array([])
    mses = np.array([])
    expected = np.array([ [1., 0.], [0., 1.], [0., 1.], [1., 0.] ])

    mse = se = 0.

    try:
        for i in range(interations):
            train_score, test_score, proba = method(X, y, nodes)
            # print proba
            scores = np.append(scores, test_score)
            se   = (expected[:,0] - proba[:,0])**2
            mse  = np.mean( se )
            mses = np.append(mses, mse)
            # print "training score ", train_score, " testing score ", test_score, " mean squared error: ", mse

    except KeyboardInterrupt:
        pass


    print "mean: ", np.mean(scores), " std: ", np.std(scores), " mse: ", np.mean(mses), " std: ", np.std(mses)


LOOPCOUNT = 30


filename = "data/xor.dat"
X, y = load_data(filename)

print "tanh: 1000 interations, 2 hidden nodes"
run(LOOPCOUNT, tanh, X, y, 2)
print "tanh: 1000 interations, 4 hidden nodes"
run(LOOPCOUNT, tanh, X, y, 4)
print "tanh: 200 interations, 2 nodes, 2 pieces"
run(LOOPCOUNT, maxout, X, y, 2)



# --------------
