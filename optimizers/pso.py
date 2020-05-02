from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utils import logger as lg
import copy
import math
import numpy as np
import random
random.seed(42)  # Seed the random number generator


class PSO(Optimizer):
    def __init__(self, cfg, prb):
        Optimizer.__init__(self, cfg, prb)

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
        self.prb.budget['remaining'] = self.prb.budget['total']
        self.swarm()
        return self.global_best.fitness, self.global_best.perm, self.fitness_trend

    def swarm(self):
        generator = self.cfg.settings['opt']['PSO']['generator']
        self.number_genes = len(getattr(self.prb, 'generator_' + generator)(self.pos_min, self.pos_max))
        self.initial_candidate_size = self.number_genes * 2
        lg.msg(logging.DEBUG, 'Swarm size to {}'.format(self.initial_candidate_size))

        for i in range(self.initial_candidate_size):
            candidate = Particle()
            
            # Generate perm of cont values within domain bounds
            candidate.perm_cont = getattr(self.prb, 'generator_' + generator)(self.pos_min, self.pos_max)
            
            # Transform perm of cont values back to discrete job id's using smallest position value method
            candidate.perm = self.transform_cont_perm(candidate)
            
            # Calculate fitness based on discrete jobs ids perm
            candidate.fitness = self.prb.evaluator(candidate.perm)
            
            # Set random velocity
            candidate.velocity = [round(self.velocity_min + (self.velocity_max - self.velocity_min) * 
                                        random.uniform(0, 1), 2) for j in range(self.number_genes)]

            self.archive_local_best(candidate)
            self.population.append(candidate)

        # Sort population of candidates by fitness ascending to get best (minimization)
        self.population.sort(key=lambda x: x.fitness, reverse=False)
        self.set_global_best(self.population[0])

        while self.prb.budget['remaining'] > 0:
            for ci, candidate in enumerate(self.population):
                candidate.fitness = self.prb.evaluator(candidate.perm)

                # Evaluate fitness and set personal (local) best
                if candidate.fitness < candidate.local_best_fitness:
                    self.archive_local_best(candidate)

            # Determine the current global best i.e. swarm leader
            self.population.sort(key=lambda x: x.fitness, reverse=False)

            # Update leader in swarm
            if self.population[0].fitness < self.global_best.fitness:
                self.set_global_best(self.population[0])
                self.fitness_trend.append(self.global_best.fitness)
                lg.msg(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.global_best.fitness, self.population[0].fitness))

            for ci, candidate in enumerate(self.population):                
                self.velocity(candidate)  # Update velocity of each candidate

            self.perturb_perm()

    @staticmethod
    def archive_local_best(candidate):
        candidate.local_best_fitness = candidate.fitness
        candidate.local_best_perm = candidate.perm
        candidate.local_best_perm_cont = candidate.perm_cont

    def set_global_best(self, candidate):
        self.global_best.fitness = candidate.fitness
        self.global_best.perm = candidate.perm
        self.global_best.perm_cont = candidate.perm_cont


    @staticmethod
    def transform_cont_perm(particle):
        # Get smallest position value
        spv = sorted(range(len(particle.perm_cont)), key=lambda i: particle.perm_cont[i], reverse=False)
        return spv

    def perturb_perm(self):
        for ci, candidate in enumerate(self.population):
            if ci == 0:
                continue
            for ji, j in enumerate(candidate.perm):
                candidate.perm_cont[ji] += candidate.velocity[ji]
            candidate.perm = self.transform_cont_perm(candidate)

    def velocity(self, particle):
        for pi, p in enumerate(particle.perm_cont):
            exp_inertia = particle.perm_cont[pi] + self.weight * (particle.perm_cont[pi] - particle.local_best_perm_cont[pi])
            exp_local = self.local_c1 * random.random() * (particle.local_best_perm_cont[pi] - particle.perm_cont[pi])
            exp_global = self.global_c2 * random.random() * (self.global_best.perm_cont[pi] - particle.perm_cont[pi])
            particle.velocity[pi] = exp_inertia + exp_local + exp_global

    def clamp(self, n):
        return max(min(self.velocity_max, n), self.velocity_min)
