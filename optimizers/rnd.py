from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import random
random.seed(42)  # Seed the random number generator


class RND(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

    def optimize(self):
        # Evaluating initial temperature has a computational cost, so reduce budget
        self.random_search()
        return self.gbest.fitness, self.gbest.candidate, self.fitness_trend

    def random_search(self):
        self.gbest.candidate = getattr(self.problem, 'generator_' + self.cfg.settings['opt']['RND']['generator'])(self.problem.n)
        self.gbest.fitness, self.budget = self.problem.evaluator(self.gbest.candidate, self.budget)

        while self.budget > 0:
            new = Particle()
            new.candidate = getattr(self.problem, 'generator_' + self.cfg.settings['opt']['RND']['generator'])(self.problem.n)
            new.fitness, self.budget = self.problem.evaluator(new.candidate, self.budget)

            if new.fitness < self.gbest.fitness:
                lg.msg(logging.DEBUG, 'Previous best {} replaced by new best {}'.format(self.gbest.fitness,
                                                                                        new.fitness))
                self.gbest.fitness = new.fitness
                self.gbest.candidate = new.candidate
                self.fitness_trend.append(self.gbest.fitness)
