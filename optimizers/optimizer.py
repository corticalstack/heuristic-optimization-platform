import logging
from utilities import logger as lg
import struct
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

    def binary_to_float(self, binary):
        # Transform bit string to float
        float_val = float(int(''.join([str(i) for i in binary]), 2))

        # Rescale float within lower and upper bounds of
        float_val = float_val / (2 ** self.hj.bit_computing - 1) * (self.hj.pid_ub - self.hj.pid_lb) + self.hj.pid_lb
        return float_val

    def n_swap(self, candidate):
        # This does a local search by swapping two random jobs
        new_candidate = candidate.copy()
        idx = self.random.sample(range(0, len(new_candidate)), 2)

        # Pair swap
        new_candidate[idx[0]], new_candidate[idx[1]] = new_candidate[idx[1]], new_candidate[idx[0]]
        return new_candidate

    def n_insert(self):
        pass

    def n_exchange(self, candidate):
        """
        Exchange job positions and i and j
        """
        ops = self.random.sample(range(0, len(candidate)), 2)
        candidate[ops[0]], candidate[ops[1]] = candidate[ops[1]], candidate[ops[0]]
        return candidate

    def n_shift(self):
        pass

    def n_sbox(self):
        pass

    def crossover(self, parent0, parent1):
        crossover_point = self.random.randint(1, (len(parent0) - 1))
        if self.hj.type == 'combinatorial':
            child = parent0[:crossover_point]
            for c in parent1:
                if c not in child:
                    child.append(c)
        else:
            child = parent0[:crossover_point] + parent1[crossover_point:]
        return child

    def pre_processing(self):
        # self.hj.fitness_trend = []
        #
        # # Set global best single particle if passed
        # #if 'gbest' in kwargs:
        # #    self.gbest = kwargs['gbest']
        # #else:
        # self.hj.gbest = Particle()
        #
        # # Set population of particles if passed
        # #if 'pop' in kwargs:
        # #    self.population = kwargs['pop']
        # #else:
        # self.hj.population = []
        #
        # if self.hj.initial_sample:
        #     self.hj.pid_cls.initial_sample = self.hj.pid_cls.generate_initial_sample()

        pass

    def post_processing(self):
        lg.msg(logging.DEBUG, 'Computational budget remaining is {}'.format(self.hj.budget))

