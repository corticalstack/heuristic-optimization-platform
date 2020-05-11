from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import copy


class SHC(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

    def optimize(self):
        # Evaluating initial temperature has a computational cost, so reduce budget
        self.stochastic_hill_climbing()

    def stochastic_hill_climbing(self):
        self.hj.rbest.candidate = self.hj.generator(lb=self.hj.pid_lb, ub=self.hj.pid_lb)
        self.hj.rbest.fitness, self.hj.budget = self.hj.pid_cls.evaluator(self.hj.rbest.candidate, self.hj.budget)

        while self.hj.budget > 0:
            new = Particle()
            new.candidate = self.n_swap(self.hj.rbest.candidate)  # One neighbour at random by simple pairswap
            new.fitness, self.hj.budget = self.hj.pid_cls.evaluator(new.candidate, self.hj.budget)

            if new.fitness < self.hj.rbest.fitness:
                lg.msg(logging.DEBUG, 'Previous best {} replaced by new best {}'.format(self.hj.rbest.fitness,
                                                                                        new.fitness))
                self.hj.rbest = copy.deepcopy(new)
                self.hj.rft.append(self.hj.rbest.fitness)
