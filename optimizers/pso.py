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
        self.pos_min = 0
        self.pos_max = 4
        self.velocity_clip = (-4, 4)

        self.weight = 0.5 # Inertia
        self.local_c1 = 2.1
        self.global_c2 = 2.1

        self.velocity_max = 4
        self.velocity_min = -4

    @staticmethod
    def archive_lbest(candidate):
        candidate.local_best_fitness = candidate.fitness
        candidate.local_best_perm = candidate.candidate
        candidate.local_best_perm_cont = candidate.candidate_cont

    def optimize(self):
        self.swarm()

    def swarm(self):
        self.initial_candidate_size = self.problem.n * 2
        lg.msg(logging.DEBUG, 'Swarm size to {}'.format(self.initial_candidate_size))

        # Incoming population migrates to starting population, reset velocity and continuous permutation values
        if self.population:
            self.reset_inherited_population_attr()

        # Complete assembly of initial population size, accounting for any incoming migrant population
        generator = self.cfg.settings['opt']['PSO']['generator']
        for i in range(self.initial_candidate_size - len(self.population)):
            c = Particle()
            
            # Generate candidate of cont values within domain bounds
            c.candidate_cont = getattr(self.problem, 'generator_' + generator)(self.problem.n, self.pos_min, self.pos_max)
            
            # Transform candidate of cont values back to discrete job id's using smallest position value method
            c.candidate = self.problem.candidate_spv_continuous_to_discrete(c.candidate_cont)
            
            # Calculate fitness based on discrete jobs ids perm
            c.fitness, self.budget = self.problem.evaluator(c.candidate, self.budget)
            
            # Set random velocity
            c.velocity = [round(self.velocity_min + (self.velocity_max - self.velocity_min) *
                                        self.random.uniform(0, 1), 2) for j in range(self.problem.n)]

            self.archive_lbest(c)
            self.population.append(c)

        # Sort population of candidates by fitness ascending to get best (minimization)
        self.population.sort(key=lambda x: x.fitness, reverse=False)
        self.set_gbest(self.population[0])

        while self.budget > 0:
            for ci, c in enumerate(self.population):
                c.fitness, self.budget = self.problem.evaluator(c.candidate, self.budget)

                # Evaluate fitness and set personal (local) best
                if c.fitness < c.local_best_fitness:
                    self.archive_lbest(c)

            # Determine the current global best i.e. swarm leader
            self.population.sort(key=lambda x: x.fitness, reverse=False)

            # Update leader in swarm
            if self.population[0].fitness < self.gbest.fitness:
                lg.msg(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.gbest.fitness, self.population[0].fitness))

                self.set_gbest(self.population[0])
                self.fitness_trend.append(self.gbest.fitness)

            for ci, c in enumerate(self.population):
                self.velocity(c)  # Update velocity of each candidate

            self.perturb_candidate()

    def reset_inherited_population_attr(self):
        for c in self.population:
            c.velocity = [round(self.velocity_min + (self.velocity_max - self.velocity_min) *
                                self.random.uniform(0, 1), 2) for j in range(self.problem.n)]
            c.candidate_cont = self.problem.candidate_spv_discrete_to_continuous(c.candidate, self.pos_min, self.pos_max)

            self.archive_lbest(c)

    def set_gbest(self, candidate):
        self.gbest.fitness = candidate.fitness
        self.gbest.candidate = candidate.candidate
        self.gbest.candidate_cont = candidate.candidate_cont

    def perturb_candidate(self):
        for ci, c in enumerate(self.population):
            if ci == 0:
                continue
            for ji, j in enumerate(c.candidate):
                c.candidate_cont[ji] += c.velocity[ji]
            c.candidate = self.problem.candidate_spv_continuous_to_discrete(c.candidate_cont)

    def velocity(self, particle):
        for pi, p in enumerate(particle.candidate_cont):
            exp_inertia = particle.candidate_cont[pi] + self.weight * (particle.candidate_cont[pi] - particle.local_best_perm_cont[pi])
            exp_local = self.local_c1 * self.random.random() * (particle.local_best_perm_cont[pi] - particle.candidate_cont[pi])
            exp_global = self.global_c2 * self.random.random() * (self.gbest.candidate_cont[pi] - particle.candidate_cont[pi])
            particle.velocity[pi] = exp_inertia + exp_local + exp_global

    def clamp(self, n):
        return max(min(self.velocity_max, n), self.velocity_min)
