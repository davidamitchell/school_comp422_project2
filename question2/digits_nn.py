# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

from sklearn import datasets, cross_validation
from sknn.mlp import Classifier, Layer, Convolution

from sklearn.neighbors import KNeighborsClassifier

import sys
import logging
import random

logging.basicConfig(
            # format="%(message)s",
            level=logging.ERROR,
            stream=sys.stdout)


import pandas as pd
import numpy as np

def load_data(filename):
    dt = pd.read_table(filename, header=None, sep=" ")
    # X = dt.iloc[:,:-1].values.astype(float).reshape(1000,7,7)
    X = dt.iloc[:,:-1].values.astype(float)
    y = dt.iloc[:, -1].values
    return X, y


def split_test(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)
    return X_train, X_test, y_train, y_test



class Neighbor :

    NUM_NEIGHBORS = 10
    WEIGHTS = 'uniform'

    @classmethod
    def asstring(self):
        return "Neighbor"


    @classmethod
    def structure(self):
        struc  = self.asstring() + " structure:"
        struc += "\nNumber of neighbors: " + str(self.NUM_NEIGHBORS)
        struc += "\nDistance measure: Euclidean" # Minkowski with a p of 2
        struc += "\nDistance weighted: " + str(self.WEIGHTS)

        return struc

    @classmethod
    def classify(self, X, y):
        X_train, X_test, y_train, y_test = split_test(X, y)

        clf = KNeighborsClassifier(n_neighbors = self.NUM_NEIGHBORS)
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        return train_score, test_score


class Standard :

    VALIDATION_SIZE = 0.10
    STABLE = 0.001
    LEARNING_RATE = 0.02

    @classmethod
    def asstring(self):
        return "Standard"


    @classmethod
    def layers(self):
        l = [
            Layer("Tanh", units=10),
            Layer("Tanh", units=10),
            Layer("Softmax")]
        return l

    @classmethod
    def structure(self):
        struc  = self.asstring() + " structure:"
        struc += "\nValidation size: " + str(self.VALIDATION_SIZE)
        struc += "\nLearning rate: " + str(self.LEARNING_RATE)
        struc += "\nLayers: ["
        l = self.layers()
        for layer in l:
            struc += "\n  Type: " + str(layer.type) + ", Units: " + str(layer.units)

        struc += "\n]"

        return struc

    @classmethod
    def classify(self, X, y):

        seed = random.randint(0, sys.maxint)
        X_train, X_test, y_train, y_test = split_test(X, y)

        nn = Classifier(
            layers=self.layers(),
            learning_rate = self.LEARNING_RATE,
            valid_size    = self.VALIDATION_SIZE,
            n_stable      = 10,
            f_stable      = self.STABLE,
            random_state  = seed,
            verbose       = False,
            debug         = False
        )

        nn.fit(X_train, y_train)

        train_score = nn.score(X_train, y_train)
        test_score = nn.score(X_test, y_test)

        return train_score, test_score



class Convolution :

    VALIDATION_SIZE = 0.10
    STABLE = 0.001
    LEARNING_RATE = 0.02


    @classmethod
    def asstring(self):
        return "Convolution"

    @classmethod
    def layers(self):
        l = [
            # Convolution('Rectifier', channels=8, kernel_shape=(3, 3), border_mode='full'),
            # Convolution('Rectifier', channels=5, kernel_shape=(3, 3), border_mode='full'),
            # Convolution('Rectifier', channels=8, kernel_shape=(3, 3), border_mode='valid'),
            # Layer('Rectifier', units=64),
            Layer("Maxout", units=10, pieces=2),
            # Layer("Tanh", units=10),
            Layer('Softmax')
            ]
        return l

    @classmethod
    def structure(self):
        struc  = self.asstring() + " structure:"
        struc += "\nValidation size: " + str(self.VALIDATION_SIZE)
        struc += "\nLearning rate: " + str(self.LEARNING_RATE)
        struc += "\nLayers: ["
        l = self.layers()
        for layer in l:
            struc += "\n  Type: " + str(layer.type) + ", Units: " + str(layer.units) + ", Pieces: " + str(layer.pieces)

        struc += "\n]"

        return struc



    @classmethod
    def classify(self, X, y):

        seed = random.randint(0, sys.maxint)

        X_train, X_test, y_train, y_test = split_test(X, y)

        nn = Classifier(
            layers=self.layers(),
            learning_rate = self.LEARNING_RATE,
            valid_size    = self.VALIDATION_SIZE,
            n_stable      = 10,
            f_stable      = self.STABLE,
            random_state  = seed,
            verbose       = False,
            debug         = False
        )

        nn.fit(X_train, y_train)

        train_score = nn.score(X_train, y_train)
        test_score = nn.score(X_test, y_test)

        return train_score, test_score


def run(filename, interations, klass):
    # print("\n", "="*80 )

    print("%s %-12s" % (filename, klass.asstring()), end=" " )

    X, y = load_data(filename)
    scores = np.array([])

    try:
        for i in range(interations):
            train_score, test_score = klass.classify(X, y)
            scores = np.append(scores, test_score)

            # print('training score ', train_score, ' testing score', test_score)

    except KeyboardInterrupt:
        pass

    # print('mean: ', np.mean(scores), ' std: ', np.std(scores))

    print("%-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )

LOOPCOUNT = 30
print("")
print("="*80)
print(Neighbor.structure())
print('')
print(Standard.structure())
print('')
print(Convolution.structure())
print("="*80)

print("Running for ", LOOPCOUNT, " interations\n")
files = [
        'data/digits10',
        'data/digits15',
        'data/digits30',
        'data/digits60'
    ]

for f in files:
    run(f, LOOPCOUNT, Neighbor)
    run(f, LOOPCOUNT, Standard)
    run(f, LOOPCOUNT, Convolution)


















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
