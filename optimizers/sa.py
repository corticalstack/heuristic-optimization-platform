from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import math
import copy


class SA(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

        # Optimizer specific
        self.temp = 0
        self.temp_threshold = 1
        lg.msg(logging.DEBUG, 'Temperature threshold set to {}'.format(self.temp_threshold))

        self.initial_temp = 0
        self.initial_temp_cost = 0

        self.cooling_rate = 0.99
        lg.msg(logging.DEBUG, 'Cooling rate set to {}'.format(self.cooling_rate))

    def optimize(self):
        self.initial_temp = self.set_initial_temp()
        lg.msg(logging.DEBUG, 'Initial temperature set to {}'.format(self.initial_temp))
        self.anneal()
        # Evaluating initial temperature has a one-time computational cost, so reduce budget if required
        if self.initial_temp_cost != 0:
            self.hj.budget += self.initial_temp_cost
            self.initial_temp_cost = 0

    def anneal(self):
        # Set initial solution candidate
        if self.hj.rbest.fitness == self.hj.rbest.fitness_default:
            self.hj.rbest.candidate = self.get_generator()(lb=self.hj.pid_lb, ub=self.hj.pid_ub)
            self.hj.rbest.fitness, self.hj.budget = self.hj.pid_cls.evaluator(self.hj.rbest.candidate, self.hj.budget)

        self.temp = self.initial_temp

        while self.hj.budget > 0:
            if self.temp < self.temp_threshold:
                if self.hj.reheat:
                    self.temp = self.initial_temp
                else:
                    break

            new_p = Particle()

            # If continuous problem generate new solution otherwise perturb current candidate combination
            if self.hj.type == 'continuous':
                new_p.candidate = self.get_generator()(lb=self.hj.pid_lb, ub=self.hj.pid_ub)
            else:
                new_p.candidate = self.hj.variator(self.hj.rbest.candidate)

            new_p.fitness, self.hj.budget = self.hj.pid_cls.evaluator(new_p.candidate, self.hj.budget)
            loss = self.hj.rbest.fitness - new_p.fitness
            # if loss > 0.7:
            #     loss = 0.7
            probability = math.exp(loss / self.temp)

            if (new_p.fitness < self.hj.rbest.fitness) or (self.random.random() < probability):
                lg.msg(logging.DEBUG, 'Previous best {} replaced by new best {}'.format(self.hj.rbest.fitness,
                                                                                        new_p.fitness))
                self.hj.rbest = copy.deepcopy(new_p)
                self.hj.rft.append(self.hj.rbest.fitness)
                self.hj.ili.append(self.hj.budget_total - self.hj.budget)

            self.temp *= self.cooling_rate

        lg.msg(logging.DEBUG, 'Completed annealing with temperature at {}'.format(self.temp))

    def set_initial_temp(self):
        candidates = []
        for candidate in self.hj.pid_cls.initial_sample:
            fitness, self.initial_temp_cost = self.hj.pid_cls.evaluator(candidate, self.initial_temp_cost)
            candidates.append(fitness)

        # Initial temperature set to 20% of temperature spread of sample
        it = int((max(candidates) - min(candidates)) / 5)
        # import numpy as np
        # it = int(np.percentile(candidates, 95))
        return it
