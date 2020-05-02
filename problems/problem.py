import logging
from utils.visualisation import Visualisation


class Problem:
    """
    Problem super class
    """
    def __init__(self):
        self.vis = Visualisation()
        self.logger = logging.getLogger()
        self.budget = {'total': 0, 'remaining': 0}
