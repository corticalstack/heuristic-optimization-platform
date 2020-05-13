from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import copy
from operator import attrgetter
import math
import numpy as np


class PSO(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

        # Optimizer specific
        self.gbest_swarm = []
        self.prev_swarm = []

        self.velocity_min = -self.hj.oid_ub
        self.velocity_max = self.hj.oid_ub

    @staticmethod
    def clamp(candidate):
        new_candidate = []
        for c in candidate:
            new_candidate.append(max(min(5.12, c), -5.12))
        return new_candidate

    def optimize(self):
        self.swarm()

    def swarm(self):
        # Incoming population migrates to starting population, reset velocity and continuous permutation values
        if self.hj.population:
            self.reset_inherited_population_attr()

        # Complete assembly of initial population size, accounting for any incoming migrant population
        for i in range(self.hj.initial_candidate_size - len(self.hj.population)):
            new_c = Particle()
            
            # Generate candidate of cont values within domain bounds
            new_c.candidate_cont = self.hj.generator(lb=self.hj.oid_lb, ub=self.hj.oid_ub)

            if self.hj.type == 'combinatorial':
                # Transform candidate of cont values back to discrete using smallest position value method
                new_c.candidate = self.hj.pid_cls.candidate_spv_continuous_to_discrete(new_c.candidate_cont)
            else:
                new_c.candidate = new_c.candidate_cont

            new_c.fitness, self.hj.budget = self.hj.pid_cls.evaluator(new_c.candidate, self.hj.budget)
            
            self.hj.population.append(new_c)

        self.gbest_swarm = copy.deepcopy(self.hj.population)
        self.prev_swarm = copy.deepcopy(self.hj.population)
        self.set_rbest(min(self.hj.population, key=attrgetter('fitness')))

        while self.hj.budget > 0:
            new_swarm = self.swarm_in_motion()
            self.prev_swarm = copy.deepcopy(self.hj.population)
            self.hj.population = copy.deepcopy(new_swarm)

            # Evaluate population fitness and set personal (local) best
            for ci, c in enumerate(self.hj.population):
                c.fitness, self.hj.budget = self.hj.pid_cls.evaluator(c.candidate, self.hj.budget)
                if c.fitness < self.gbest_swarm[ci].fitness:
                    self.gbest_swarm[ci] = copy.deepcopy(c)
                if c.fitness < self.hj.rbest.fitness:
                    self.set_rbest(c)
                    self.hj.rft.append(c.fitness)

    def reset_inherited_population_attr(self):
        pass
        # for c in self.hj.population:
        #     c.velocity = [round(self.velocity_min + (self.velocity_max - self.velocity_min) *
        #                         self.random.uniform(0, 1), 2) for j in range(self.hj.pid_cls.n)]
        #     c.candidate_cont = self.hj.pid_cls.candidate_spv_discrete_to_continuous(c.candidate, self.hj.pid_lb, self.hj.pid_ub)
        #
        #     self.gbest_pop_lbest(c)

    def set_rbest(self, candidate):
        self.hj.rbest = copy.deepcopy(candidate)

    def swarm_in_motion(self):
        new_s = []
        for ci, c in enumerate(self.hj.population):
            new_c = Particle()  # New candidate particle
            new_c.candidate_cont = []
            for pi, p in enumerate(c.candidate_cont):
                exp_inertia = p + self.hj.coeff_inertia * (p - self.prev_swarm[ci].candidate_cont[pi])
                exp_local = self.hj.coeff_local * self.random.random() * (self.gbest_swarm[ci].candidate_cont[pi] - p)
                exp_global = self.hj.coeff_global * self.random.random() * (self.hj.rbest.candidate_cont[pi] - p)
                velocity = exp_inertia + exp_local + exp_global
                new_c.candidate_cont.append(velocity)

            if self.hj.type == 'combinatorial':
                new_c.candidate = self.hj.pid_cls.candidate_spv_continuous_to_discrete(new_c.candidate_cont)
            else:
                new_c.candidate_cont = self.clamp(new_c.candidate_cont)
                new_c.candidate = new_c.candidate_cont
            new_s.append(new_c)

        return new_s

