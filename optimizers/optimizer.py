from optimizers.particle import Particle
import logging
from utils import logger as lg


class Optimizer:
    def __init__(self, cfg, prb):
        # Persist current configuration and problem
        self.cfg = cfg
        self.prb = prb

        self.logger = logging.getLogger()
        self.initial_candidate_size = 1
        self.global_best = None
        self.population = []
        self.fitness_trend = []

    def before_start(self):
        self.global_best = Particle()
        self.population = []
        self.fitness_trend = []

    def on_completion(self):
        lg.msg(logging.DEBUG, 'Computational budget remaining is {}'.format(self.prb.budget['remaining']))
