from optimizers.particle import Particle
import logging
from utils import logger as lg


class Optimizer:
    def __init__(self, random, cfg, prb):
        # Persist current configuration and problem
        self.random = random
        self.cfg = cfg
        self.prb = prb

        self.logger = logging.getLogger()
        self.initial_candidate_size = 1
        self.global_best = None
        self.population = []
        self.fitness_trend = []

    def new_neighbour_pairswap(self, perm):
        # This does a local search by swapping two random jobs
        new_perm = perm.copy()
        idx = self.random.sample(range(0, len(new_perm)), 2)

        # Pair swap
        new_perm[idx[0]], new_perm[idx[1]] = new_perm[idx[1]], new_perm[idx[0]]
        return new_perm

    def before_start(self):
        self.global_best = Particle()
        self.population = []
        self.fitness_trend = []

    def on_completion(self):
        lg.msg(logging.DEBUG, 'Computational budget remaining is {}'.format(self.prb.budget['remaining']))

