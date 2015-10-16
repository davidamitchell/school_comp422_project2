# -*- coding: utf-8 -*-
from __future__ import (absolute_import, unicode_literals, print_function)

from time import time
import random
import sys
import inspyred
import numpy as np
import math
import pylab

LOWER_BOUND = -30.0
UPPER_BOUND =  30.0

def generator_20(random, args):
    dim = 20
    # return [ random.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(dim) ]
    return [ random.randint(LOWER_BOUND, UPPER_BOUND) for _ in range(dim) ]

def generator_50(random, args):
    dim = 50
    # return [ random.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(dim) ]
    return [ random.randint(LOWER_BOUND, UPPER_BOUND) for _ in range(dim) ]


class GriewankPSO :

    POP_SIZE          = 100
    MAX_EVALUATIONS   = 20000
    INERTIA           = 1.0
    CONGNITIVE_RATE   = 1.0
    SOCIAL_RATE       = 1.5
    TOPOLOGY          = 'star'
    NEIGHBORHOOD_SIZE = 10

    @classmethod
    def asstring(self):
        return "Griewank"

    @classmethod
    def structure(self):
        struc  = self.asstring() + " structure:"
        struc += "\nPopulation size: " + str(self.POP_SIZE) + " Number of evaluations: " + str(self.MAX_EVALUATIONS)
        struc += "\nCongnitive rate: " + str(self.CONGNITIVE_RATE) + " Social rate: " + str(self.SOCIAL_RATE)
        struc += "\nInertia: " + str(self.INERTIA)
        struc += "\nTopology: " + str(self.TOPOLOGY)

        if self.TOPOLOGY != 'star':
            struc += " Neighborhood size: " + str(self.NEIGHBORHOOD_SIZE)

        return struc

    @classmethod
    def griewank(self, D):
        sum = 0.
        product = 1.
        result = 0.

        for i, xi in enumerate(D):
            sum     += xi**2
            product *= ( math.cos( xi / math.sqrt( i+1. ) ) )
            result   =  ( sum / 4000. ) - product + 1.

        return result


    @classmethod
    def evaluation(self, candidates, args):
        fitness = []
        for c in candidates:
            fitness.append( self.griewank(c) )

        return fitness


    @classmethod
    def optimise(self, generator):
        prng = random.Random()
        prng.seed(random.randint(0, sys.maxint))

        ea = inspyred.swarm.PSO(prng)

        ea.terminator = inspyred.ec.terminators.evaluation_termination
        if self.TOPOLOGY == 'star':
            ea.topology = inspyred.swarm.topologies.star_topology
        else:
            ea.topology = inspyred.swarm.topologies.ring_topology

        final_pop = ea.evolve(generator         = generator,
                              evaluator         = self.evaluation,
                              bounder           = inspyred.ec.Bounder(LOWER_BOUND-1, UPPER_BOUND),
                              pop_size          = self.POP_SIZE,
                              max_evaluations   = self.MAX_EVALUATIONS,
                              congnitive_rate   = self.CONGNITIVE_RATE,
                              social_rate       = self.SOCIAL_RATE,
                              maximize          = False,
                              inertia           = 0.5
                              )


        #
        best = max(final_pop)

        # print('Best Solution: \n{0}'.format(str(best)))
        return best, best.fitness


class RosenbrockPSO :

    POP_SIZE          = 100
    MAX_EVALUATIONS   = 50000
    INERTIA           = 0.5
    CONGNITIVE_RATE   = 1.0
    SOCIAL_RATE       = 1.1
    TOPOLOGY          = 'star'
    NEIGHBORHOOD_SIZE = 10


    @classmethod
    def asstring(self):
        return "Rosenbrock"

    @classmethod
    def structure(self):
        struc  = self.asstring() + " structure:"
        struc += "\nPopulation size: " + str(self.POP_SIZE) + " Number of evaluations: " + str(self.MAX_EVALUATIONS)
        struc += "\nCongnitive rate: " + str(self.CONGNITIVE_RATE) + " Social rate: " + str(self.SOCIAL_RATE)
        struc += "\nInertia: " + str(self.INERTIA)
        struc += "\nTopology: " + str(self.TOPOLOGY)

        if self.TOPOLOGY != 'star':
            struc += " Neighborhood size: " + str(self.NEIGHBORHOOD_SIZE)

        return struc

    @classmethod
    def rosenbrock(self, D):
        sum = 0.0
        for i in range(len(D) - 1):
            x  = D[ i   ]
            x1 = D[ i+1 ]
            a  = x**2 - x1
            b  = x - 1.

            sum += 100. * a**2 + b**2
        return sum


    @classmethod
    def evaluation(self, candidates, args):
        fitness = []
        for c in candidates:
            fitness.append( self.rosenbrock(c) )

        return fitness


    @classmethod
    def optimise(self, generator):
        problem = inspyred.benchmarks.Rosenbrock(30)
        prng = random.Random()
        prng.seed(random.randint(0, sys.maxint))

        ea = inspyred.swarm.PSO(prng)

        ea.terminator = inspyred.ec.terminators.evaluation_termination
        if self.TOPOLOGY == 'star':
            ea.topology = inspyred.swarm.topologies.star_topology
        else:
            ea.topology = inspyred.swarm.topologies.ring_topology

        final_pop = ea.evolve(generator         = generator,
                              evaluator         = self.evaluation,
                              bounder           = inspyred.ec.Bounder(LOWER_BOUND, UPPER_BOUND),
                              pop_size          = self.POP_SIZE,
                              max_evaluations   = self.MAX_EVALUATIONS,
                              congnitive_rate   = self.CONGNITIVE_RATE,
                              social_rate       = self.SOCIAL_RATE,
                              maximize          = False,
                              inertia           = self.INERTIA
                              )


        #
        best = max(final_pop)

        # print('Best Solution rosenbrock: \n{0}'.format(str(best)))
        return best, best.fitness


#

def run(interations, klass, generator):
    scores = np.array([])
    print("%-12s:" % (klass.asstring()), end=" " )
    try:
        for i in range(interations):
            solution, fitness = klass.optimise(generator)
            scores = np.append(scores, fitness)
            # print('fitness: ', fitness)

    except KeyboardInterrupt:
        pass

    print("%-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )


LOOPCOUNT = 10
print('')
print('='*80)
print(RosenbrockPSO.structure())
# print('')
# print(GriewankPSO.structure())
# print('')


print("Running for ", LOOPCOUNT, " interations")
print("Bounded by (", LOWER_BOUND, ", ",  UPPER_BOUND, ")")
print("20 Dimensions")
run(LOOPCOUNT, RosenbrockPSO, generator_20)
run(LOOPCOUNT, GriewankPSO, generator_20)
print("50 Dimensions")
run(LOOPCOUNT, RosenbrockPSO, generator_50)
run(LOOPCOUNT, GriewankPSO, generator_50)
