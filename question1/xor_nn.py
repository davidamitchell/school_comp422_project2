import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sknn.mlp import Classifier, Layer

import sys
import logging

logging.basicConfig(
            # format="%(message)s",
            level=logging.DEBUG,
            stream=sys.stdout)

def load_data():
    dt = pd.read_table("data/xor.dat", header=None, sep=" ")
    X = dt.iloc[:,:-1].as_matrix()
    y = dt.iloc[:, -1].as_matrix()
    return X, y

def load_test():
    dt = pd.read_table("data/xor_test.dat", header=None, sep=" ")
    X = dt.iloc[:,:-1].as_matrix()
    y = dt.iloc[:, -1].as_matrix()
    return X, y

def standard():

    X_train, y_train = load_data()
    X_test, y_test = load_test()

    nn = Classifier(
        layers=[
            Layer("Tanh", units=2),
            Layer("Softmax")],
        learning_rate=0.2,
        # n_iter=10000,
        n_stable=200,
        f_stable=0.001,
        valid_size=0.25,
        verbose=True,
        debug=True
        )

    try:
        nn.fit(X_train, y_train)
    except KeyboardInterrupt:
        pass

    print "standard 10x10 tanh score: ", nn.score(X_test, y_test)
    print
    print confusion_matrix(y_test, nn.predict(X_test))
    print y_test
    print nn.predict(X_test).tolist()
    print nn.predict_proba(X_test).tolist()
    # print nn.predict(np.array([1,1]))
    # print nn.predict(np.array([0,0]))
    # print nn.predict(np.array([1,0]))
    # print nn.predict(np.array([0,1]))


standard()






# --------------
