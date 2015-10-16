from time import time
from random import Random
import inspyred

LOWER_BOUND = -30.0
UPPER_BOUND =  30.0

def generator_20(random, args):
    dim = 20
    # return [ random.uniform(LOWER_BOUND, UPPER_BOUND) for _ in range(dim) ]
    return [ random.randint(LOWER_BOUND, UPPER_BOUND) for _ in range(dim) ]

def main(prng=None, display=False):
    if prng is None:
        prng = Random()
        prng.seed(time())

    problem = inspyred.benchmarks.Rosenbrock(30)
    ea = inspyred.swarm.PSO(prng)
    ea.terminator = inspyred.ec.terminators.evaluation_termination
    # ea.topology = inspyred.swarm.topologies.ring_topology
    ea.topology = inspyred.swarm.topologies.star_topology
    final_pop = ea.evolve(generator=generator_20,
                          evaluator=problem.evaluator,
                          pop_size=100,
                        #   bounder=problem.bounder,
                        #   bounder=inspyred.ec.Bounder([LOWER_BOUND-1] * 20, [UPPER_BOUND] * 20),
                          bounder           = inspyred.ec.Bounder(LOWER_BOUND, UPPER_BOUND),
                          maximize=False,
                          max_evaluations=100000,
                          neighborhood_size=5)

    if display:
        best = max(final_pop)
        print('Best Solution: \n{0}'.format(str(best)))
    return ea

if __name__ == '__main__':
    main(display=True)
