from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utils import logger as lg
import math
import numpy as np
import random
random.seed(42)  # Seed the random number generator


class GA(Optimizer):
    def __init__(self, cfg, prb):
        Optimizer.__init__(self, cfg, prb)

        # Optimizer specific
        self.parents = []
        self.children = []

        self.number_parents = 3
        lg.msg(logging.DEBUG, 'Number of parents set to {}'.format(self.number_parents))

        self.number_children = 5
        lg.msg(logging.DEBUG, 'Number of children set to {}'.format(self.number_children))

        self.population = []
        self.number_genes = 0

    def optimize(self):
        self.prb.budget['remaining'] = self.prb.budget['total']
        self.evolve()
        return self.global_best.fitness, self.global_best.perm, self.fitness_trend

    def evolve(self):
        self.initial_candidate_size = len(getattr(self.prb, 'generator_' +
                                                  self.cfg.settings['opt']['GA']['generator'])()) * 2
        lg.msg(logging.DEBUG, 'Initial candidate size set to {}'.format(self.initial_candidate_size))

        for i in range(self.initial_candidate_size):
            candidate = Particle()
            candidate.perm = getattr(self.prb, 'generator_' + self.cfg.settings['opt']['GA']['generator'])()
            self.population.append(candidate)

        if len(self.population) > 0:
            self.number_genes = len(self.population[0].perm)

        while self.prb.budget['remaining'] > 0:
            for ci, candidate in enumerate(self.population):
                if candidate.fitness == candidate.fitness_default:
                    candidate.fitness = self.prb.evaluator(candidate.perm)

            # Sort population by fitness ascending
            self.population.sort(key=lambda x: x.fitness, reverse=False)

            if self.population[0].fitness < self.global_best.fitness:
                lg.msg(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.global_best.fitness, self.population[0].fitness))
                self.global_best.fitness = self.population[0].fitness
                self.global_best.perm = self.population[0].perm
                self.fitness_trend.append(self.population[0].fitness)

            self.parents = self.parent_selection()

            self.children = self.parent_crossover()

            self.children_mutate()

            self.population = self.update_population()

    def update_population(self):
        new_pop = []
        for p in self.parents:
            particle = Particle()
            particle.fitness = self.population[p].fitness
            particle.perm = self.population[p].perm
            new_pop.append(particle)

        # Add children to population
        for c in self.children:
            particle = Particle()
            particle.perm = c
            new_pop.append(particle)

        return new_pop

    def parent_selection(self):
        # Fitness proportionate selection (FPS), assigning probabilities to individuals acting as parents depending on their
        # fitness
        max_fitness = sum([particle.fitness for particle in self.population])
        fitness_proportionate = [particle.fitness / max_fitness for particle in self.population]

        pointer_distance = 1 / self.number_parents
        start_point = random.uniform(0, pointer_distance)
        points = [start_point + i * pointer_distance for i in range(self.number_parents)]

        # Add boundary points
        points.insert(0, 0)
        points.append(1)

        parents = []

        fitness_aggr = 0
        for fi, fp in enumerate(fitness_proportionate):
            if len(parents) == self.number_parents:
                break
            fitness_aggr += fp
            for pi, p in enumerate(points):
                if p < fitness_aggr < points[pi+1]:
                    parents.append(fi)
                    points.pop(0)
                    break

        return parents

    def parent_crossover(self):
        children = []
        for i in range(self.number_children):
            crossover_point = random.randint(1, self.number_genes - 1)
            child = self.population[self.parents[0]].perm[:crossover_point]
            for c in self.population[self.parents[1]].perm:
                if c not in child:
                    child.append(c)
            children.append(child)

        return children

    def children_mutate(self):
        """
        Swap 2 tasks at random
        """
        # Swap positions of the 2 job tasks in the candidate
        for i in range(self.number_children):
            # Generate 2 task numbers at random, within range
            tasks = random.sample(range(0, self.number_genes), 2)
            self.children[i][tasks[0]], self.children[i][tasks[1]] = \
                self.children[i][tasks[1]], self.children[i][tasks[0]]
