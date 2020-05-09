from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg


class SHC(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

    def optimize(self):
        # Evaluating initial temperature has a computational cost, so reduce budget
        self.stochastic_hill_climbing()
        return self.gbest.fitness, self.gbest.candidate, self.fitness_trend

    def stochastic_hill_climbing(self):
        self.gbest.candidate = getattr(self.problem, 'generator_' + self.cfg.settings['opt']['SHC']['generator'])(self.problem.n)
        self.gbest.fitness, self.budget = self.problem.evaluator(self.gbest.candidate, self.budget)

        while self.budget > 0:
            new = Particle()
            new.candidate = self.n_swap(self.gbest.candidate)  # One neighbour at random by simple pairswap
            new.fitness, self.budget = self.problem.evaluator(new.candidate, self.budget)

            if new.fitness < self.gbest.fitness:
                lg.msg(logging.DEBUG, 'Previous best {} replaced by new best {}'.format(self.gbest.fitness,
                                                                                        new.fitness))
                self.gbest.fitness = new.fitness
                self.gbest.candidate = new.candidate
                self.fitness_trend.append(self.gbest.fitness)
