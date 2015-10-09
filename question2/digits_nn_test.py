# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

from sklearn import datasets, cross_validation
from sknn.mlp import Classifier, Layer, Convolution


import sys
import logging

logging.basicConfig(
            # format="%(message)s",
            level=logging.ERROR,
            stream=sys.stdout)


import pandas as pd
import numpy as np
#
# # Load the data and split it into subsets for training and testing.
# digits = datasets.load_digits()
# X = digits.images
# y = digits.target
#
# filename = 'data/digits40'

def load_data(filename):
    dt = pd.read_table(filename, header=None, sep=" ")
    # X = dt.iloc[:,:-1].values.astype(float)
    X = dt.iloc[:,:-1].values.astype(float).reshape(1000,7,7)
    y = dt.iloc[:, -1].values
    return X, y

# print(X[0])
# print(X[0].shape)
# print(X.shape)
# X = X.reshape(1000,7,7)
# print(X.shape)
# print(y.shape)
# X = y.reshape(1000,7,7)
# print(X.shape)

# print(X)
# exit()




def split_test(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)
    return X_train, X_test, y_train, y_test



def convolution(X, y):

    X_train, X_test, y_train, y_test = split_test(X, y)
    nn = Classifier(
        layers=[
            Convolution('Rectifier', channels=5, kernel_shape=(3, 3), border_mode='full'),
            # Convolution('Rectifier', channels=5, kernel_shape=(3, 3), border_mode='full'),
            # Convolution('Rectifier', channels=8, kernel_shape=(3, 3), border_mode='valid'),
            # Layer('Rectifier', units=64),
            Layer('Softmax')],
        learning_rate=0.002,
        valid_size=0.2,
        n_stable=10,
        verbose=False,
        debug=False)

    nn.fit(X_train, y_train)

    train_score = nn.score(X_train, y_train)
    test_score = nn.score(X_test, y_test)

    return train_score, test_score


def run(filename):
    print("\n", "="*80 )
    print( filename )

    X, y = load_data(filename)
    scores = np.array([])

    for i in range(30):
        train_score, test_score = convolution(X, y)
        scores = np.append(scores, test_score)
        print('training score', train_score, 'testing score', test_score)

    print('mean: ', np.mean(scores), ' std: ', np.std(scores))

    #


# run('data/digits00')
run('data/digits15')
run('data/digits30')
run('data/digits60')

#
# y_pred = nn.predict(X_test)
#
#
# # Show some training images and some test images too.
# import matplotlib.pyplot as pylab
#
# # for index, (image, label) in enumerate(zip(digits.images[:10], digits.target[:10])):
# #     pylab.subplot(2, 6, index + 1)
# #     pylab.axis('off')
# #     pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
# #     pylab.title('Training: %i' % label)
#
# for index, (image, label) in enumerate(zip(X[:10], y[:10])):
#     pylab.subplot(2, 10, index + 1)
#     pylab.axis('off')
#     pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
#     pylab.title('Training: %i' % label)
# #
# for index, (image, prediction) in enumerate(zip(X_test[:10], y_pred[:10])):
#     pylab.subplot(2, 10, index + 11)
#     pylab.axis('off')
#     pylab.imshow(image.reshape((7,7)), cmap=pylab.cm.gray_r, interpolation='nearest')
#     pylab.title('Predicts: %i' % prediction)

# pylab.show()
