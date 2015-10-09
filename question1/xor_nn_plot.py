import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sknn.mlp import Classifier, Layer

import matplotlib.pyplot as plt

def load_data():
    dt = pd.read_table("data/xor.dat", header=None, sep=" ")
    X = dt.iloc[:,:-1].as_matrix()
    y = dt.iloc[:, -1].as_matrix()
    return X, y



def standard():

    X_train, y_train = load_data()
    X_test, y_test = load_data()
    X, y = load_data()

    title = "Learning Curves"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10,
                                       test_size=0.25, random_state=0)

    nn = Classifier(
        layers=[
            Layer("Tanh", units=2),
            Layer("Softmax")],
        learning_rate=0.2,
        n_iter=100,
        # n_stable=10,
        # f_stable=0.2
        )



    # nn.fit(X_train, y_train)

    plot_learning_curve(nn, title, X, y, cv=cv, n_jobs=4, train_sizes=4)

    # print "standard 10x10 tanh score: ", nn.score(X_test, y_test)
    # print
    # print confusion_matrix(y_test, nn.predict(X_test))
    # print y_test
    # print nn.predict(X_test).tolist()
    # print nn.predict_proba(X_test).tolist()
    # print nn.predict(np.array([1,1]))
    # print nn.predict(np.array([0,0]))
    # print nn.predict(np.array([1,0]))
    # print nn.predict(np.array([0,1]))


standard()


plt.show()




# --------------
