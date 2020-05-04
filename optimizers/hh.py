from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utils import logger as lg
import math
import numpy as np
from importlib import import_module


class HH(Optimizer):
    def __init__(self, random, cfg, prb):
        Optimizer.__init__(self, random, cfg, prb)

        # Optimizer specific
        self.low_level_heuristics = {}

    def optimize(self):
        self.prb.budget['remaining'] = self.prb.budget['total']
        self.import_low_level_heuristics()
        self.hyper()
        return self.global_best.fitness, self.global_best.perm, self.fitness_trend

    def import_low_level_heuristics(self):
        try:
            for hc in self.cfg.settings['opt'][self.prb.oid]['low_level_components']:
                my_module = import_module('optimizers.' + hc.lower())
                cls = getattr(my_module, hc)(self.random, self.cfg, self.prb)
                self.low_level_heuristics[hc] = {'cls': cls, 'trend': []}
        except (ModuleNotFoundError, AttributeError) as e:
            lg.msg(logging.INFO, 'HH error {} importing low-level heuristic models'.format(e))

    def hyper(self):
        budget_remaining = self.prb.budget['remaining']
        while budget_remaining > 0:
            for llh in self.low_level_heuristics:
                # Override generic computational budget
                self.cfg.settings['gen']['comp_budget_base'] = 10
                self.prb.set_budget()
                self.low_level_heuristics[llh]['cls'].before_start()
                candidate = Particle()
                candidate.fitness, candidate.perm, run_ft = self.low_level_heuristics[llh]['cls'].optimize()
                budget_remaining -= self.prb.budget['total']
                self.low_level_heuristics[llh]['trend'].append(candidate)

