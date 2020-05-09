from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import math
import numpy as np


class SA(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

        # Optimizer specific
        self.temp = 0
        self.temp_threshold = 1
        lg.msg(logging.DEBUG, 'Temperature threshold set to {}'.format(self.temp_threshold))

        self.initial_temp_cost = 0
        self.initial_temp_weight = 0.035
        self.initial_temp = self.set_initial_temp()  # JP juist a thought - set initial temp as tghe spread - (max - min)
        lg.msg(logging.DEBUG, 'Initial temperature set to {}'.format(self.initial_temp))

        self.cooling_rate = 0.99
        lg.msg(logging.DEBUG, 'Cooling rate set to {}'.format(self.cooling_rate))

    def optimize(self):
        self.anneal()
        # Evaluating initial temperature has a one-time computational cost, so reduce budget if required
        if self.initial_temp_cost != 0:
            self.budget += self.initial_temp_cost
            self.initial_temp_cost = 0

    def anneal(self):
        if self.gbest.fitness == self.gbest.fitness_default:
            self.gbest.candidate = getattr(self.problem, 'generator_' + self.cfg.settings['opt']['SA']['generator'])(self.problem.n)
            self.gbest.fitness, self.budget = self.problem.evaluator(self.gbest.candidate, self.budget)
        self.temp = self.initial_temp

        while self.budget > 0 and (self.temp > self.temp_threshold):
            new = Particle()
            new.candidate = self.n_swap(self.gbest.candidate)

            new.fitness, self.budget = self.problem.evaluator(new.candidate, self.budget)
            loss = self.gbest.fitness - new.fitness
            probability = math.exp(loss / self.temp)

            rr = self.random.random()
            if (new.fitness < self.gbest.fitness) or (rr < probability):
                lg.msg(logging.DEBUG, 'Previous best {} replaced by new best {}'.format(self.gbest.fitness,
                                                                                        new.fitness))
                if rr < probability:
                    lg.msg(logging.DEBUG, 'Random {} less than probability {}'.format(rr, probability))
                self.gbest.fitness = new.fitness
                self.gbest.candidate = new.candidate
                self.fitness_trend.append(self.gbest.fitness)

            self.temp *= self.cooling_rate

        lg.msg(logging.DEBUG, 'Completed annealing with temperature at {}'.format(self.temp))

    def set_initial_temp(self):
        candidates = []
        for candidate in self.problem.initial_sample:
            fitness, self.initial_temp_cost = self.problem.evaluator(candidate, self.initial_temp_cost)
            candidates.append(fitness)

        it = int(np.mean(candidates)) * self.initial_temp_weight
        lg.msg(logging.INFO, 'Initial temperature set to {}'.format(it))
        return it
