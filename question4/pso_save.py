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
    result = 0.

    for i, xi in enumerate(D):

        sum     += xi**2
        product *= ( math.cos( xi / math.sqrt( i+1. ) ) )

        result =  ( sum / 4000. ) - product + 1.

    return result

def griewank_evaluation(candidates, args):
    fitness = []
    for c in candidates:
        fitness.append( griewank(c) )

    return fitness

# def griewank_evaluation(candidates, args):
#     fitness = []
#     for c in candidates:
#         prod = 1
#         for i, x in enumerate(c):
#             prod *= math.cos(x / math.sqrt(i+1))
#         fitness.append(1.0 / 4000.0 * sum([x**2 for x in c]) - prod + 1)
#     return fitness

def generator(random, args):
    dim = 20
    return [ random.uniform(-30., 30.) for _ in range(dim) ]

from time import time
import random
import sys
import inspyred


SEED = random.randint(0, sys.maxint)

def griewank_run():
    prng = random.Random()
    prng.seed(SEED)
    # prng.seed(3592626484087809527)

    ea = inspyred.swarm.PSO(prng)

    stats_observer = inspyred.ec.observers.stats_observer
    stats_observer.num_generations = 100

    ea.terminator = inspyred.ec.terminators.evaluation_termination
    ea.topology = inspyred.swarm.topologies.ring_topology
    # ea.topology = inspyred.swarm.topologies.star_topology
    # ea.observer = stats_observer

    #
    # inertia - the inertia constant to be used in the particle updating (default 0.5)
    # cognitive_rate - the rate at which the particles current position influences its movement (default 2.1)
    # social_rate - the rate at which the particles neighbors influence its movement (default 2.1)

    final_pop = ea.evolve(generator         = generator,
                          evaluator         = griewank_evaluation,
                          bounder           = inspyred.ec.Bounder(-30, 30),
                          pop_size          = 100,
                          maximize          = False,
                          max_evaluations   = 10000,
                          neighborhood_size = 10,
                          congnitive_rate   = 0.2,
                          social_rate       = 1.0)
    #
    # inspyred.ec.observers.stats_observer(population, num_generations, num_evaluations, args)
    # inspyred.ec.observers.file_observer(population, num_generations, num_evaluations, args)


    best = max(final_pop)
    print('Best Solution griewank: \n{0}'.format(str(best)))

    # inspyred.ec.analysis.generation_plot("output.png", errorbars=False)

Class RosenbrockPSO :

    @classmethod
    def optimise():
        prng = random.Random()
        prng.seed(SEED)
        # prng.seed(880295265843339639)

        ea = inspyred.swarm.PSO(prng)

        # stats_observer = inspyred.ec.observers.stats_observer
        # stats_observer.num_generations = 10000

        ea.terminator = inspyred.ec.terminators.evaluation_termination
        # ea.topology = inspyred.swarm.topologies.ring_topology
        ea.topology = inspyred.swarm.topologies.star_topology
        # ea.observer = stats_observer

        #
        # inertia - the inertia constant to be used in the particle updating (default 0.5)
        # cognitive_rate - the rate at which the particles current position influences its movement (default 2.1)
        # social_rate - the rate at which the particles neighbors influence its movement (default 2.1)

        final_pop = ea.evolve(generator         = generator,
                              evaluator         = rosenbrock_evaluation,
                              bounder           = inspyred.ec.Bounder(-30, 30),
                              pop_size          = 200,
                              maximize          = False,
                              max_evaluations   = 10000,
                            #   neighborhood_size = 10,
                              congnitive_rate   = 1.0,
                              social_rate       = 1.0)
        #
        # inspyred.ec.observers.stats_observer(population, num_generations, num_evaluations, args)
        # inspyred.ec.observers.file_observer(population, num_generations, num_evaluations, args)


        best = max(final_pop)
        print('Best Solution rosenbrock: \n{0}'.format(str(best)))

        # inspyred.ec.analysis.generation_plot("output.png", errorbars=False)



scores = np.array([])

try:
    for i in range(INTERATIONS):
        # griewank_run()
        rosenbrock_run()
        # print_summary(hof, toolbox)
        mse = get_mse(hof[0], toolbox)
        scores = np.append(scores, mse)
        print 'mse ', mse, ' exp: ', hof[0]

except KeyboardInterrupt:
    pass

# print('mean: ', np.mean(scores), ' std: ', np.std(scores))

print("%-6s %2.5f %-5s %2.5f" % ('mean: ', np.mean(scores), ' std: ', np.std(scores)) )
print "seed: ", SEED



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
