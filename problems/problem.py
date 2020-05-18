import logging
from utilities.visualisation import Visualisation
import numpy as np
import pandas as pd

class Problem:
    """
    Problem super class
    """
    def __init__(self, **kwargs):
        self.random = kwargs['random']
        self.hj = kwargs['hopjob']

        self.vis = Visualisation()
        self.logger = logging.getLogger()
        self.n = 0

    @staticmethod
    def candidate_spv_continuous_to_discrete(c):
        # Get smallest position value
        spv = sorted(range(len(c)), key=lambda i: c[i], reverse=False)
        return spv

    def pre_processing(self):
        pass  # Placeholder

    def post_processing(self):
        pass  # Placeholder

    def generator_discrete(self, **kwargs):
        candidate = list(range(0, self.n))
        np.random.shuffle(candidate)
        return candidate

    def candidate_spv_discrete_to_continuous(self, *args):
        candidate, pos_min, pos_max = args
        cont = []
        linspace = np.linspace(pos_min, pos_max, self.n, endpoint=True)
        for ci, c in enumerate(candidate):
            if self.hj.pid_type == 'combinatorial':
                cont.append(round(linspace[c], 2))
            else:
                cont.append(round(linspace[ci], 2))
        return cont

    def generator_chromosome(self, **kwargs):
        chromosome = []
        for _ in range(self.n):
            chromosome.append([self.random.choice([0, 1]) for _ in range(self.hj.bit_computing)])
        return chromosome

    def generator_continuous(self, **kwargs):
        candidate = []
        for _ in range(self.n):
            candidate.append(self.random.uniform(kwargs['lb'], kwargs['ub']))
        return candidate

    def generate_initial_sample(self):
        sample = []
        num = int(self.n * (self.hj.budget * self.hj.sample_size_coeff))

        # Default sample size to 10 in case above results in 0, typically during testing framework with small budget
        if num == 0:
            num = 10
        for _ in range(num):
            if self.hj.pid_type == 'combinatorial':
                sample.append(self.hj.generator_comb(lb=self.hj.pid_lb, ub=self.hj.pid_ub))
            else:
                sample.append(self.hj.generator_cont(lb=self.hj.pid_lb, ub=self.hj.pid_ub))
        return sample
