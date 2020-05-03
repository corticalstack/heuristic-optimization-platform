import logging
from utils.visualisation import Visualisation


class Problem:
    """
    Problem super class
    """
    def __init__(self, random):
        self.random = random
        self.vis = Visualisation()
        self.logger = logging.getLogger()
        self.n_dimensions = 0
        self.budget = {'total': 0, 'remaining': 0}

    @staticmethod
    def perm_spv_continuous_to_discrete(c):
        # Get smallest position value
        spv = sorted(range(len(c)), key=lambda i: c[i], reverse=False)
        return spv

