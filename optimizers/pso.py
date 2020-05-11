from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import copy
import math
import numpy as np


class PSO(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

        # Optimizer specific        
        self.weight = 0.5  # Inertia
        self.local_c1 = 2.1
        self.global_c2 = 2.1

        self.velocity_min = -self.hj.oid_ub
        self.velocity_max = self.hj.oid_ub

    @staticmethod
    def archive_lbest(candidate):
        candidate.lbest_fitness = candidate.fitness
        candidate.lbest_perm = candidate.candidate
        candidate.lbest_perm_cont = candidate.candidate_cont

    def optimize(self):
        self.swarm()

    def swarm(self):
        self.initial_candidate_size = self.hj.pid_cls.n * 2
        lg.msg(logging.DEBUG, 'Swarm size to {}'.format(self.initial_candidate_size))

        # Incoming population migrates to starting population, reset velocity and continuous permutation values
        if self.hj.population:
            self.reset_inherited_population_attr()

        # Complete assembly of initial population size, accounting for any incoming migrant population
        for i in range(self.initial_candidate_size - len(self.hj.population)):
            c = Particle()
            
            # Generate candidate of cont values within domain bounds
            samples = self.hj.pid_cls.n ** 2
            c.candidate_cont = self.hj.generator(n=samples, lb=self.hj.oid_lb, ub=self.hj.oid_ub)
            
            # Transform candidate of cont values back to discrete job id's using smallest position value method
            c.candidate = self.hj.pid_cls.candidate_spv_continuous_to_discrete(c.candidate_cont)
            
            # Calculate fitness based on discrete jobs ids perm
            c.fitness, self.hj.budget = self.hj.pid_cls.evaluator(c.candidate, self.hj.budget)
            
            # Set random velocity
            c.velocity = [round(self.velocity_min + (self.velocity_max - self.velocity_min) *
                                self.random.uniform(0, 1), 2) for j in range(self.hj.pid_cls.n)]

            self.archive_lbest(c)
            self.hj.population.append(c)

        # Sort population of candidates by fitness ascending to get best (minimization)
        self.hj.population.sort(key=lambda x: x.fitness, reverse=False)
        self.set_rbest(self.hj.population[0])

        while self.hj.budget > 0:
            # Evaluate population fitness and set personal (local) best
            for ci, c in enumerate(self.hj.population):
                c.fitness, self.hj.budget = self.hj.pid_cls.evaluator(c.candidate, self.hj.budget)

                if c.fitness < c.lbest_fitness:
                    self.archive_lbest(c)

            # Determine the current global best i.e. swarm leader
            self.hj.population.sort(key=lambda x: x.fitness, reverse=False)

            # Update leader in swarm as this run best
            if self.hj.population[0].fitness < self.hj.rbest.fitness:
                lg.msg(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.hj.rbest.fitness, self.hj.population[0].fitness))

                self.set_rbest(self.hj.population[0])
                self.hj.rft.append(self.hj.rbest.fitness)

            for ci, c in enumerate(self.hj.population):
                self.velocity(c)  # Update velocity of each candidate

            self.perturb_candidate()

    def reset_inherited_population_attr(self):
        for c in self.hj.population:
            c.velocity = [round(self.velocity_min + (self.velocity_max - self.velocity_min) *
                                self.random.uniform(0, 1), 2) for j in range(self.hj.pid_cls.n)]
            c.candidate_cont = self.hj.pid_cls.candidate_spv_discrete_to_continuous(c.candidate, self.hj.pid_lb, self.hj.pid_ub)

            self.archive_lbest(c)

    def set_rbest(self, candidate):
        self.hj.rbest = copy.deepcopy(candidate)

    def perturb_candidate(self):
        for ci, c in enumerate(self.hj.population):
            if ci == 0:
                continue
            for ji, j in enumerate(c.candidate):
                c.candidate_cont[ji] += c.velocity[ji]
            c.candidate = self.hj.pid_cls.candidate_spv_continuous_to_discrete(c.candidate_cont)

    def velocity(self, particle):
        for pi, p in enumerate(particle.candidate_cont):
            exp_inertia = particle.candidate_cont[pi] + self.weight * (particle.candidate_cont[pi] - particle.lbest_perm_cont[pi])
            exp_local = self.local_c1 * self.random.random() * (particle.lbest_perm_cont[pi] - particle.candidate_cont[pi])
            exp_global = self.global_c2 * self.random.random() * (self.hj.rbest.candidate_cont[pi] - particle.candidate_cont[pi])
            particle.velocity[pi] = exp_inertia + exp_local + exp_global

    def clamp(self, n):
        return max(min(self.velocity_max, n), self.velocity_min)
