import logging
from utilities.visualisation import Visualisation
import numpy as np


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

    def pre_processing(self):
        pass  # Placeholder

    def post_processing(self):
        pass  # Placeholder

    def generator_discrete(self, **kwargs):
        candidate = list(range(0, self.n))
        np.random.shuffle(candidate)
        return candidate

    @staticmethod
    def candidate_spv_continuous_to_discrete(c):
        # Get smallest position value
        spv = sorted(range(len(c)), key=lambda i: c[i], reverse=False)
        return spv

    def candidate_spv_discrete_to_continuous(self, *args):
        candidate, pos_min, pos_max = args
        cont = []
        linspace = np.linspace(pos_min, pos_max, self.n, endpoint=False)
        for c in candidate:
            cont.append(round(linspace[c], 2))
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
        num = int(self.n * self.hj.sample_size_factor)
        for _ in range(num):
            sample.append(self.hj.generator(lb=self.hj.pid_lb, ub=self.hj.pid_ub))
        return sample
