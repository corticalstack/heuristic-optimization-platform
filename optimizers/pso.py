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

        self.weight = 0.5  # Inertia
        self.local_c1 = 2.1
        self.global_c2 = 2.1

        self.velocity_min = -self.hj.oid_ub
        self.velocity_max = self.hj.oid_ub

    def optimize(self):
        self.swarm()
        self.hj.rbest.candidate = self.hj.rbest.candidate_cont

    def swarm(self):
        # Incoming population migrates to starting population, reset velocity and continuous permutation values
        if self.hj.population:
            self.reset_inherited_population_attr()

        # Complete assembly of initial population size, accounting for any incoming migrant population
        for i in range(self.hj.initial_candidate_size - len(self.hj.population)):
            c = Particle()
            
            # Generate candidate of cont values within domain bounds
            samples = self.hj.pid_cls.n ** 2
            c.candidate_cont = self.hj.generator(n=samples, lb=self.hj.oid_lb, ub=self.hj.oid_ub)

            if self.hj.type == 'combinatorial':
                # Transform candidate of cont values back to discrete using smallest position value method
                c.candidate = self.hj.pid_cls.candidate_spv_continuous_to_discrete(c.candidate_cont)
            else:
                c.candidate = c.candidate_cont

            c.fitness, self.hj.budget = self.hj.pid_cls.evaluator(c.candidate, self.hj.budget)
            
            self.hj.population.append(c)

        self.gbest_swarm = copy.deepcopy(self.hj.population)
        self.prev_swarm = copy.deepcopy(self.hj.population)
        self.set_rbest(min(self.hj.population, key=attrgetter('fitness')))

        while self.hj.budget > 0:
            new_swarm = self.swarm_in_motion()
            self.prev_swarm = copy.deepcopy(self.hj.population)
            self.hj.population = copy.deepcopy(new_swarm)

            # Evaluate population fitness and set personal (local) best
            for ci, c in enumerate(self.hj.population):
                c.fitness, self.hj.budget = self.hj.pid_cls.evaluator(c.candidate_cont, self.hj.budget)
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
        ns = []
        for ci, c in enumerate(self.hj.population):
            nc = Particle()  # New candidate particle
            nc.candidate_cont = []
            for pi, p in enumerate(c.candidate_cont):
                exp_inertia = p + self.weight * (p - self.prev_swarm[ci].candidate_cont[pi])
                exp_local = self.local_c1 * self.random.random() * (self.gbest_swarm[ci].candidate_cont[pi] - p)
                exp_global = self.global_c2 * self.random.random() * (self.hj.rbest.candidate_cont[pi] - p)
                velocity = exp_inertia + exp_local + exp_global
                nc.candidate_cont.append(velocity)

            if self.hj.type == 'combinatorial':
                nc.candidate_cont = self.hj.pid_cls.candidate_spv_continuous_to_discrete(nc.candidate_cont)
            else:
                nc.candidate_cont = self.clamp(nc.candidate_cont)
            ns.append(nc)

        return ns

    def clamp(self, candidate):
        new_candidate = []
        for c in candidate:
            new_candidate.append(max(min(5.12, c), -5.12))
        return new_candidate
