import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sknn.mlp import Classifier, Layer

import sys
import logging

logging.basicConfig(
            # format="%(message)s",
            level=logging.ERROR,
            stream=sys.stdout)


import random

def load_data(filename):
    dt = pd.read_table(filename, header=None, sep=" ")
    X = dt.iloc[:,:-1].as_matrix()
    y = dt.iloc[:, -1].as_matrix()
    return X, y

def load_test():
    dt = pd.read_table("data/xor_test.dat", header=None, sep=" ")
    X = dt.iloc[:,:-1].as_matrix()
    y = dt.iloc[:, -1].as_matrix()
    return X, y
#
# seed = random.randint(0, sys.maxint)
# seed = 5683336684240711446
def standard(X, y):

    seed = random.randint(0, sys.maxint)

    X_train, y_train = X, y
    X_test, y_test = X, y

    nn = Classifier(
        layers=[
            Layer("Maxout", units=2, pieces=2),
            Layer("Softmax")],
        learning_rate=0.2,
        batch_size=1,
        n_iter=100,
        random_state=seed,
        verbose=False,
        debug=False
        )

    # print nn.layers[0]

    nn.fit(X_train, y_train)



    # #
    # print
    # # print nn.__getstate__()
    # # print np.array(nn.__getstate__()['weights'][0][1]).shape
    # print nn.__getstate__()['weights']
    # print

    train_score = nn.score(X_train, y_train)
    test_score = nn.score(X_test, y_test)
    proba = nn.predict_proba(X_test)


    # print 'seed: ', seed
    # print confusion_matrix(y_test, nn.predict(X_test))
    # print y_test
    # print nn.predict(X_test).tolist()
    # print nn.predict_proba(X_test).tolist()
    # print nn.predict(np.array([1,1]))
    # print nn.predict(np.array([0,0]))
    # print nn.predict(np.array([1,0]))
    # print nn.predict(np.array([0,1]))
    return train_score, test_score, proba



filename = "data/xor.dat"
X, y = load_data(filename)
scores = np.array([])
mse = np.array([])
expected = np.array([ [1., 0.], [0., 1.], [0., 1.], [1., 0.] ])

se = 0.

try:
    for i in range(30):
        train_score, test_score, proba = standard(X, y)
        scores = np.append(scores, test_score)

        # print ( (expected - proba)**2 ).tolist()
        # print np.sum( (expected - proba)**2 )

        se = np.sum( (expected - proba)**2 ) / 2.
        mse = np.append(mse, se)
        print('training score', train_score, 'testing score', test_score, ' squared error: ', se, ')

except KeyboardInterrupt:
    pass

# print("Seed: ", seed)
print('mean: ', np.mean(scores), ' std: ', np.std(scores), ' mse: ', np.mean(mse), ' std: ', np.std(mse))





# --------------
