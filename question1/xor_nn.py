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

    nn.fit(X_train, y_train)

    train_score = nn.score(X_train, y_train)
    test_score = nn.score(X_test, y_test)
    proba = nn.predict_proba(X_test)

    return train_score, test_score, proba



filename = "data/xor.dat"
X, y = load_data(filename)
scores = np.array([])
mse = np.array([])
expected = np.array([ [1., 0.], [0., 1.], [0., 1.], [1., 0.] ])

se = 0.

try:
    for i in range(3):
        train_score, test_score, proba = standard(X, y)
        scores = np.append(scores, test_score)

        se = np.sum( (expected[:,0] - proba[:,0])**2 )
        mse = np.append(mse, se)
        print "training score ", train_score, " testing score ", test_score, " squared error: ", se

except KeyboardInterrupt:
    pass


print "mean: ", np.mean(scores), " std: ", np.std(scores), " mse: ", np.mean(mse), " std: ", np.std(mse)





# --------------
