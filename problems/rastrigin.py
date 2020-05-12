from problems.problem import Problem
import logging
from utilities import logger as lg
from utilities.stats import Stats
import math
import numpy as np


class RASTRIGIN(Problem):
    """
    Rastrigin
    """
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)

        # Set n dimensions
        self.n = 2

        # Set computational budget scaled to problem instance dimensions
        self.set_budget()

    def post_processing(self, **kwargs):
        pass

    def set_budget(self):
        pass
        # Base budget * problem dimensions
        #self.budget['total'] = self.cfg.settings['gen']['comp_budget_base'] * self.n
        #self.budget['remaining'] = self.budget['total']

    def evaluator(self, candidate, budget=1):
        fitness = []
        budget -=1
        return sum([x**2 - 10 * math.cos(2 * math.pi * x) + 10 for x in candidate]), budget
