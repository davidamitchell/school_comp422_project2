import numpy as np
import math
import pylab

def rosenbrock(D):
    sum = 0.0
    for i in range(len(D) - 1):
        x  = D[ i   ]
        x1 = D[ i+1 ]
        a  = x**2 - x1
        b  = x - 1.

        sum += 100. * a**2 + b**2
    return sum


def rosenbrock_evaluation(candidates, args):
    fitness = []
    for c in candidates:
        fitness.append( rosenbrock(c) )

    return fitness

def griewank(D):
    sum = 0.
    product = 1.

    for i, xi in enumerate(D):

        sum     += xi**2
        product *= ( math.cos( xi / math.sqrt( i+1. ) ) )

        result =  ( sum / 4000. ) - product + 1.

def griewank_evaluation(candidates, args):
    fitness = []
    for c in candidates:
        fitness.append( griewank(c) )

    return fitness

def generator(random, args):
    dim = 20
    return [ random.uniform(-30., 30.) for _ in range(dim) ]

from time import time
from random import Random
import inspyred

def main():
    prng = Random()
    prng.seed(time())

    ea = inspyred.swarm.PSO(prng)

    ea.terminator = inspyred.ec.terminators.evaluation_termination
    # ea.topology = inspyred.swarm.topologies.ring_topology
    ea.topology = inspyred.swarm.topologies.star_topology

    final_pop = ea.evolve(generator = generator,
                          evaluator = rosenbrock_evaluation,
                          bounder = inspyred.ec.Bounder(-3, 3),
                          pop_size=100,
                          maximize=False,
                          max_evaluations=30000,
                          neighborhood_size=5)


    best = max(final_pop)
    print('Best Solution: \n{0}'.format(str(best)))

if __name__ == '__main__':
    main()



    #
    # def generator(self, random, args):
    #     return [random.uniform(-600.0, 600.0) for _ in range(self.dimensions)]
    #
    # def evaluator(self, candidates, args):
    #     fitness = []
    #     for c in candidates:
    #         prod = 1
    #         for i, x in enumerate(c):
    #             prod *= math.cos(x / math.sqrt(i+1))
    #         fitness.append(1.0 / 4000.0 * sum([x**2 for x in c]) - prod + 1)
    #     return fitness
    #
    #
    # def generator(self, random, args):
    #     return [random.uniform(-5.0, 10.0) for _ in range(self.dimensions)]
    #
    # def evaluator(self, candidates, args):
    #     fitness = []
    #     for c in candidates:
    #         total = 0
    #         for i in range(len(c) - 1):
    #             total += 100 * (c[i]**2 - c[i+1])**2 + (c[i] - 1)**2
    #         fitness.append(total)
    #     return fitness