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

        self.initial_temp = 0
        self.initial_temp_cost = 0
        self.initial_temp_weight = 0.035


        self.cooling_rate = 0.99
        lg.msg(logging.DEBUG, 'Cooling rate set to {}'.format(self.cooling_rate))

    def optimize(self):
        self.initial_temp = self.set_initial_temp()  # JP juist a thought - set initial temp as tghe spread - (max - min)
        lg.msg(logging.DEBUG, 'Initial temperature set to {}'.format(self.initial_temp))
        self.anneal()
        # Evaluating initial temperature has a one-time computational cost, so reduce budget if required
        if self.initial_temp_cost != 0:
            self.hj.budget += self.initial_temp_cost
            self.initial_temp_cost = 0

    def anneal(self):
        if self.hj.gbest.fitness == self.hj.gbest.fitness_default:
            self.hj.gbest.candidate = self.hj.generator(lb=self.hj.pid_lb, ub=self.hj.pid_lb)
            self.hj.gbest.fitness, self.hj.budget = self.hj.pid_cls.evaluator(self.hj.gbest.candidate, self.hj.budget)
        self.temp = self.initial_temp
        self.temp = 100
        while self.hj.budget > 0 and (self.temp > self.temp_threshold):
            new = Particle()
            if len(self.hj.gbest.candidate) == 1:
                new.candidate = self.hj.generator(lb=self.hj.pid_lb, ub=self.hj.pid_lb)
            else:
                new.candidate = self.n_swap(self.hj.gbest.candidate)

            new.fitness, self.hj.budget = self.hj.pid_cls.evaluator(new.candidate, self.hj.budget)
            loss = self.hj.gbest.fitness - new.fitness
            probability = math.exp(loss / self.temp)

            rr = self.random.random()
            if (new.fitness < self.hj.gbest.fitness) or (rr < probability):
                lg.msg(logging.DEBUG, 'Previous best {} replaced by new best {}'.format(self.hj.gbest.fitness,
                                                                                        new.fitness))
                if rr < probability:
                    lg.msg(logging.DEBUG, 'Random {} less than probability {}'.format(rr, probability))
                self.hj.gbest.fitness = new.fitness
                self.hj.gbest.candidate = new.candidate
                self.hj.fitness_trend.append(self.hj.gbest.fitness)

            self.temp *= self.cooling_rate

        lg.msg(logging.DEBUG, 'Completed annealing with temperature at {}'.format(self.temp))

    def set_initial_temp(self):
        candidates = []
        for candidate in self.hj.pid_cls.initial_sample:
            fitness, self.initial_temp_cost = self.hj.pid_cls.evaluator(candidate, self.initial_temp_cost)
            candidates.append(fitness)

        it = int(np.mean(candidates)) * self.initial_temp_weight
        lg.msg(logging.INFO, 'Initial temperature set to {}'.format(it))
        return it
