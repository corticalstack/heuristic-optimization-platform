from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import math
import numpy as np


class GA(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

        # Optimizer specific
        self.parents = []
        self.children = []

    def optimize(self):
        self.evolve()

    def evolve(self):
        for i in range(self.hj.initial_candidate_size):
            candidate = Particle()
            candidate.candidate = self.hj.generator(lb=self.hj.pid_lb, ub=self.hj.pid_ub)
            self.hj.population.append(candidate)

        while self.hj.budget > 0:

            # Evaluate any new candidates
            for ci, candidate in enumerate(self.hj.population):
                if candidate.fitness == candidate.fitness_default:
                    c = candidate.candidate
                    if self.hj.generator.__name__ == 'generator_chromosome':
                        c = self.binary_to_float(c)
                    candidate.fitness, self.hj.budget = self.hj.pid_cls.evaluator(c, self.hj.budget)

            # Sort population by fitness ascending
            self.hj.population.sort(key=lambda x: x.fitness, reverse=False)

            if self.hj.population[0].fitness < self.hj.rbest.fitness:
                lg.msg(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.hj.gbest.fitness, self.hj.population[0].fitness))
                self.hj.rbest.fitness = self.hj.population[0].fitness
                self.hj.rbest.candidate = self.hj.population[0].candidate
                self.hj.rft.append(self.hj.population[0].fitness)

            self.parents = self.parent_selection()

            self.children = self.parent_crossover()

            self.children_mutate()

            self.hj.population = self.update_population()

    def update_population(self):
        new_pop = []
        for p in self.parents:
            particle = Particle()
            particle.fitness = self.hj.population[p].fitness
            particle.candidate = self.hj.population[p].candidate
            new_pop.append(particle)

        # Add children to population
        for c in self.children:
            particle = Particle()
            particle.candidate = c
            new_pop.append(particle)

        return new_pop

    def parent_selection(self):
        # Fitness proportionate selection (FPS), assigning probabilities to individuals based on fitness
        max_fitness = sum([particle.fitness for particle in self.hj.population])

        #fitness_proportionate = [particle.fitness / max_fitness for particle in self.hj.population]  # For maximisation

        # Fitness proportionate where smaller fitness is better
        fitness_proportionate = [((max_fitness - particle.fitness) / max_fitness) / (len(self.hj.population) - 1) for particle in self.hj.population]

        pointer_distance = 1 / self.hj.number_parents
        start_point = self.random.uniform(0, pointer_distance)
        points = [start_point + i * pointer_distance for i in range(self.hj.number_parents)]

        # Add boundary points
        points.insert(0, 0)
        points.append(1)

        parents = []

        fitness_aggr = 0
        for fi, fp in enumerate(fitness_proportionate):
            if len(parents) == self.hj.number_parents:
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
        for i in range(self.hj.number_children):
            child = self.crossover(self.hj.population[self.parents[0]].candidate, self.hj.population[self.parents[1]].candidate)
            children.append(child)

        return children

    def children_mutate(self):
        for i in range(self.hj.number_children):
            self.children[i] = self.n_exchange(self.children[i])
