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
        self.budget = {'total': 0, 'remaining': 0}

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

    # def get_generator(self, oid):
    #     # Optimizer configured generator overrides higher level problem generator e.g. PSO works on continuous values
    #     if 'generator' in self.cfg.settings['opt'][oid]:
    #         generator = self.cfg.settings['opt'][oid]['generator']
    #     else:
    #         generator = self.cfg.settings['prb'][self.__class__.__name__]['generator']
    #     return generator

    def candidate_spv_discrete_to_continuous(self, *args):
        candidate, pos_min, pos_max = args
        cont = []
        linspace = np.linspace(pos_min, pos_max, self.n, endpoint=False)
        for c in candidate:
            cont.append(round(linspace[c], 2))
        return cont

    def generator_continuous(self, **kwargs):
        candidate = []
        for j in range(self.n):
            candidate.append(round(kwargs['lb'] + (kwargs['ub'] - kwargs['lb']) *
                                   self.random.uniform(0, 1), 2))
        return candidate

    def generate_initial_sample(self):
        sample = []
        num = int(self.n * self.hj.sample_size_factor)

        for i in range(num):
            sample.append(self.hj.generator(lb=self.hj.pid_lb, ub=self.hj.pid_ub))

        return sample
