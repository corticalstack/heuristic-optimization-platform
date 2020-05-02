from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utils import logger as lg
import math
import numpy as np
import random
random.seed(42)  # Seed the random number generator


class SA(Optimizer):
    def __init__(self, cfg, prb):
        Optimizer.__init__(self, cfg, prb)

        # Optimizer specific
        self.temp = 0
        self.temp_threshold = 1
        lg.msg(logging.DEBUG, 'Temperature threshold set to {}'.format(self.temp_threshold))

        self.initial_temp_weight = 0.035
        self.initial_temp = self.set_initial_temp()
        lg.msg(logging.DEBUG, 'Initial temperature set to {}'.format(self.initial_temp))

        self.cooling_rate = 0.99
        lg.msg(logging.DEBUG, 'Cooling rate set to {}'.format(self.cooling_rate))

    def optimize(self):
        # Evaluating initial temperature has a computational cost, so reduce budget
        self.prb.budget['remaining'] = self.prb.budget['total'] - len(self.prb.initial_sample)
        self.anneal()
        return self.global_best.fitness, self.global_best.perm, self.fitness_trend

    def anneal(self):
        self.global_best.perm = getattr(self.prb, 'generator_' + self.cfg.settings['opt']['SA']['generator'])()
        self.global_best.fitness = self.prb.evaluator(self.global_best.perm)
        self.temp = self.initial_temp

        while self.prb.budget['remaining'] > 0 and (self.temp > self.temp_threshold):
            new = Particle()
            new.perm = self.neighbour_solution()  #JP to check if get neighbour solution OK/valid

            new.fitness = self.prb.evaluator(new.perm)
            loss = self.global_best.fitness - new.fitness
            probability = math.exp(loss / self.temp)

            rr = random.random()
            if (new.fitness < self.global_best.fitness) or (rr < probability):
                lg.msg(logging.DEBUG, 'Previous best {} replaced by new best {}'.format(self.global_best.fitness,
                                                                                        new.fitness))
                if rr < probability:
                    lg.msg(logging.DEBUG, 'Random {} less than probability {}'.format(rr, probability))
                self.global_best.fitness = new.fitness
                self.global_best.perm = new.perm
                self.fitness_trend.append(self.global_best.fitness)

            self.temp *= self.cooling_rate

        lg.msg(logging.DEBUG, 'Completed annealing with temperature at {}'.format(self.temp))

    def set_initial_temp(self):
        candidates = []
        for candidate in self.prb.initial_sample:
            fitness = self.prb.evaluator(candidate)
            candidates.append(fitness)

        it = int(np.mean(candidates)) * self.initial_temp_weight
        lg.msg(logging.INFO, 'Initial temperature set to {}'.format(it))
        return it

    def neighbour_solution(self):
        # This does a local search by swapping two random jobs
        new_candidate_perm = self.global_best.perm.copy()
        tasks = random.sample(range(0, len(new_candidate_perm)), 2)
        new_candidate_perm[tasks[0]], new_candidate_perm[tasks[1]] = \
            new_candidate_perm[tasks[1]], new_candidate_perm[tasks[0]]
        return new_candidate_perm

    # Could always add alternative neighbour generators here and just take the fittest one

