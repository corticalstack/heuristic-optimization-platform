from optimizers.particle import Particle
import logging
from utilities import logger as lg


class Optimizer:
    def __init__(self, **kwargs):
        # Persist current configuration and problem
        self.random = kwargs['random']
        self.cfg = kwargs['cfg']
        self.problem = kwargs['problem']

        self.logger = logging.getLogger()
        self.initial_candidate_size = 1
        self.gbest = None
        self.population = []
        self.fitness_trend = []
        self.budget = 0

    def run(self, **kwargs):
        self.budget = kwargs['budget']
        self.pre_processing()
        self.optimize()
        self.post_processing()
        return self.gbest.fitness, self.gbest.perm, self.fitness_trend

    def n_swap(self, perm):
        # This does a local search by swapping two random jobs
        new_perm = perm.copy()
        idx = self.random.sample(range(0, len(new_perm)), 2)

        # Pair swap
        new_perm[idx[0]], new_perm[idx[1]] = new_perm[idx[1]], new_perm[idx[0]]
        return new_perm

    def n_insert(self):
        pass

    def n_exchange(self):
        pass

    def n_shift(self):
        pass

    def n_sbox(self):
        pass

    def pre_processing(self):
        self.gbest = Particle()
        self.population = []
        self.fitness_trend = []

    def post_processing(self):
        lg.msg(logging.DEBUG, 'Computational budget remaining is {}'.format(self.budget))

