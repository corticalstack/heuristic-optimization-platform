from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utils import logger as lg


class SHC(Optimizer):
    def __init__(self, random, cfg, prb):
        Optimizer.__init__(self, random, cfg, prb)

    def optimize(self):
        # Evaluating initial temperature has a computational cost, so reduce budget
        self.prb.budget['remaining'] = self.prb.budget['total']
        self.stochastic_hill_climbing()
        return self.global_best.fitness, self.global_best.perm, self.fitness_trend

    def stochastic_hill_climbing(self):
        self.global_best.perm = getattr(self.prb, 'generator_' + self.cfg.settings['opt']['SHC']['generator'])()
        self.global_best.fitness = self.prb.evaluator(self.global_best.perm)

        while self.prb.budget['remaining'] > 0:
            new = Particle()
            new.perm = self.new_neighbour_pairswap(self.global_best.perm)  # One neighbour at random by simple pairswap
            new.fitness = self.prb.evaluator(new.perm)

            if new.fitness < self.global_best.fitness:
                lg.msg(logging.DEBUG, 'Previous best {} replaced by new best {}'.format(self.global_best.fitness,
                                                                                        new.fitness))
                self.global_best.fitness = new.fitness
                self.global_best.perm = new.perm
                self.fitness_trend.append(self.global_best.fitness)
