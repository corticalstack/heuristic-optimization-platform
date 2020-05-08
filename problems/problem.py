import logging
from utilities.visualisation import Visualisation
import numpy as np


class Problem:
    """
    Problem super class
    """
    def __init__(self, **kwargs):
        self.random = kwargs['random']
        self.vis = Visualisation()
        self.logger = logging.getLogger()
        self.n = 0
        self.budget = {'total': 0, 'remaining': 0}

    def pre_processing(self):
        pass

    def post_processing(self):
        pass

    @staticmethod
    def generator_discrete(n):
        candidate = list(range(0, n))
        np.random.shuffle(candidate)
        return candidate

    @staticmethod
    def perm_spv_continuous_to_discrete(c):
        # Get smallest position value
        spv = sorted(range(len(c)), key=lambda i: c[i], reverse=False)
        return spv

    def perm_spv_discrete_to_continuous(self, *args):
        perm, pos_min, pos_max = args
        cont = []
        linspace = np.linspace(pos_min, pos_max, self.n, endpoint=False)
        for p in perm:
            cont.append(round(linspace[p], 2))
        return cont

    def generator_continuous(self, *args):
        n, pos_min, pos_max = args
        candidate = []
        for j in range(n):
            candidate.append(round(pos_min + (pos_max - pos_min) * self.random.uniform(0, 1), 2))
        return candidate
