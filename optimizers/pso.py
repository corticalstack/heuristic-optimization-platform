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
        self.number_genes = 0

        self.pos_min = 0
        self.pos_max = 4
        self.velocity_clip = (-4, 4)

        self.weight = 0.5 # Inertia
        self.local_c1 = 2.1
        self.global_c2 = 2.1

        self.velocity_max = 4
        self.velocity_min = -4

    def optimize(self):
        self.swarm()
        return self.gbest.fitness, self.gbest.perm, self.fitness_trend

    def swarm(self):
        generator = self.cfg.settings['opt']['PSO']['generator']
        self.number_genes = len(getattr(self.problem, 'generator_' + generator)(self.problem.n, self.pos_min, self.pos_max))
        self.initial_candidate_size = self.number_genes * 2
        lg.msg(logging.DEBUG, 'Swarm size to {}'.format(self.initial_candidate_size))

        for i in range(self.initial_candidate_size):
            candidate = Particle()
            
            # Generate perm of cont values within domain bounds
            candidate.perm_cont = getattr(self.problem, 'generator_' + generator)(self.problem.n, self.pos_min, self.pos_max)
            
            # Transform perm of cont values back to discrete job id's using smallest position value method
            candidate.perm = self.problem.perm_spv_continuous_to_discrete(candidate.perm_cont)
            
            # Calculate fitness based on discrete jobs ids perm
            candidate.fitness, self.budget = self.problem.evaluator(candidate.perm, self.budget)
            
            # Set random velocity
            candidate.velocity = [round(self.velocity_min + (self.velocity_max - self.velocity_min) * 
                                        self.random.uniform(0, 1), 2) for j in range(self.number_genes)]

            self.archive_lbest(candidate)
            self.population.append(candidate)

        # Sort population of candidates by fitness ascending to get best (minimization)
        self.population.sort(key=lambda x: x.fitness, reverse=False)
        self.set_gbest(self.population[0])

        while self.budget > 0:
            for ci, candidate in enumerate(self.population):
                candidate.fitness, self.budget = self.problem.evaluator(candidate.perm, self.budget)

                # Evaluate fitness and set personal (local) best
                if candidate.fitness < candidate.local_best_fitness:
                    self.archive_lbest(candidate)

            # Determine the current global best i.e. swarm leader
            self.population.sort(key=lambda x: x.fitness, reverse=False)

            # Update leader in swarm
            if self.population[0].fitness < self.gbest.fitness:
                self.set_gbest(self.population[0])
                self.fitness_trend.append(self.gbest.fitness)
                lg.msg(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.gbest.fitness, self.population[0].fitness))

            for ci, candidate in enumerate(self.population):                
                self.velocity(candidate)  # Update velocity of each candidate

            self.perturb_perm()

    @staticmethod
    def archive_lbest(candidate):
        candidate.local_best_fitness = candidate.fitness
        candidate.local_best_perm = candidate.perm
        candidate.local_best_perm_cont = candidate.perm_cont

    def set_gbest(self, candidate):
        self.gbest.fitness = candidate.fitness
        self.gbest.perm = candidate.perm
        self.gbest.perm_cont = candidate.perm_cont

    def perturb_perm(self):
        for ci, candidate in enumerate(self.population):
            if ci == 0:
                continue
            for ji, j in enumerate(candidate.perm):
                candidate.perm_cont[ji] += candidate.velocity[ji]
            candidate.perm = self.problem.perm_spv_continuous_to_discrete(candidate.perm_cont)

    def velocity(self, particle):
        for pi, p in enumerate(particle.perm_cont):
            exp_inertia = particle.perm_cont[pi] + self.weight * (particle.perm_cont[pi] - particle.local_best_perm_cont[pi])
            exp_local = self.local_c1 * self.random.random() * (particle.local_best_perm_cont[pi] - particle.perm_cont[pi])
            exp_global = self.global_c2 * self.random.random() * (self.gbest.perm_cont[pi] - particle.perm_cont[pi])
            particle.velocity[pi] = exp_inertia + exp_local + exp_global

    def clamp(self, n):
        return max(min(self.velocity_max, n), self.velocity_min)
