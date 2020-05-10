import logging
from utilities.visualisation import Visualisation
import numpy as np


class Problem:
    """
    Problem super class
    """
    def __init__(self, **kwargs):
        self.random = kwargs['random']
        self.cfg = kwargs['cfg']
        self.oid = kwargs['oid']

        if 'iid' in kwargs:
            self.iid = kwargs['iid']

        self.vis = Visualisation()
        self.logger = logging.getLogger()
        self.n = 0
        self.budget = {'total': 0, 'remaining': 0}

    def pre_processing(self):
        pass  # Placeholder

    def post_processing(self):
        pass  # Placeholder

    @staticmethod
    def generator_discrete(**kwargs):
        candidate = list(range(0, kwargs['n']))
        np.random.shuffle(candidate)
        return candidate

    @staticmethod
    def candidate_spv_continuous_to_discrete(c):
        # Get smallest position value
        spv = sorted(range(len(c)), key=lambda i: c[i], reverse=False)
        return spv

    def get_generator(self, oid):
        # Optimizer configured generator overrides higher level problem generator e.g. PSO works on continuous values
        if 'generator' in self.cfg.settings['opt'][oid]:
            generator = self.cfg.settings['opt'][oid]['generator']
        else:
            generator = self.cfg.settings['prb'][self.__class__.__name__]['generator']
        return generator

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

    def generate_initial_sample(self, oid):
        sample = []
        num = int(self.n * 100)

        for i in range(num):
            sample.append(getattr(self, 'generator_' + self.get_generator(oid))(lb=self.pos_min, ub=self.pos_max))

        return sample
