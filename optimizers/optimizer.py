import logging
from utilities import logger as lg

from optimizers.particle import Particle
import copy


class Optimizer:
    def __init__(self, **kwargs):
        # Persist current configuration and problem
        self.random = kwargs['random']
        self.hj = kwargs['hopjob']
        self.initial_candidate_size = 1

    def run(self):
        self.pre_processing()
        self.optimize()
        self.post_processing()

    def n_swap(self, candidate):
        # This does a local search by swapping two random jobs
        new_candidate = candidate.copy()
        idx = self.random.sample(range(0, len(new_candidate)), 2)

        # Pair swap
        new_candidate[idx[0]], new_candidate[idx[1]] = new_candidate[idx[1]], new_candidate[idx[0]]
        return new_candidate

    def n_insert(self):
        pass

    def n_exchange(self):
        pass

    def n_shift(self):
        pass

    def n_sbox(self):
        pass

    def pre_processing(self):
        # self.fitness_trend = []
        #
        # # Set global best single particle if passed
        # if 'gbest' in kwargs:
        #     self.gbest = kwargs['gbest']
        # else:
        #     self.gbest = Particle()
        #
        # # Set population of particles if passed
        # if 'pop' in kwargs:
        #     self.population = kwargs['pop']
        # else:
        #     self.population = []

        if self.hj.initial_sample:
            self.hj.pid_cls.initial_sample = self.hj.pid_cls.generate_initial_sample()

    def post_processing(self):
        lg.msg(logging.DEBUG, 'Computational budget remaining is {}'.format(self.hj.budget))

