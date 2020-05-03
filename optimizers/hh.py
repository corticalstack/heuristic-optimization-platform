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
        self.low_level_heuristics = []

    def optimize(self):
        self.prb.budget['remaining'] = self.prb.budget['total']
        self.selection()
        return self.global_best.fitness, self.global_best.perm, self.fitness_trend

    def selection(self):
        try:
            for hc in self.cfg.settings['opt'][self.prb.oid]['low_level_components']:
                my_module = import_module('optimizers.' + hc.lower())
                cls = getattr(my_module, hc)(self.random, self.cfg, self.prb)
                self.low_level_heuristics.append(cls)
        except (ModuleNotFoundError, AttributeError) as e:
            lg.msg(logging.INFO, 'HH error {} importing low-level heuristic models'.format(e))
            print(hc)

