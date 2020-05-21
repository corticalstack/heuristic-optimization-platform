from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import copy


class GA(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

        # Optimizer specific
        self.parents = []
        self.children = []

    def optimize(self):
        self.evolve()

    def evolve(self):
        # Incoming population migrates to starting population, reset to fit GA
        if self.hj.population:
            self.reset_inherited_population_attr()

        # Complete assembly of initial population size, accounting for any incoming migrant population
        for i in range(self.hj.initial_pop_size - len(self.hj.population)):
            candidate = Particle()
            candidate.candidate = self.get_generator()(lb=self.hj.pid_lb, ub=self.hj.pid_ub)
            self.hj.population.append(candidate)

        while self.hj.budget > 0:

            # Evaluate any new candidates
            for ci, candidate in enumerate(self.hj.population):
                if candidate.fitness == candidate.fitness_default:
                    c = copy.deepcopy(candidate.candidate)
                    if self.get_generator().__name__ == 'generator_chromosome':
                        c = self.binary_to_float(c)
                    candidate.fitness, self.hj.budget = self.hj.pid_cls.evaluator(c, self.hj.budget)

            # Sort population by fitness ascending
            self.hj.population.sort(key=lambda x: x.fitness, reverse=False)

            if self.hj.population[0].fitness < self.hj.rbest.fitness:
                lg.msg(logging.DEBUG, 'Previous best is {}, now updated with new best {}'.format(
                    self.hj.rbest.fitness, self.hj.population[0].fitness))
                self.hj.rbest.fitness = self.hj.population[0].fitness
                self.hj.rbest.candidate = self.hj.population[0].candidate
                self.hj.rft.append(self.hj.population[0].fitness)
                if not self.fromhyper:
                    self.hj.iter_last_imp[self.hj.run] = self.hj.budget_total - self.hj.budget
                    self.hj.imp_count[self.hj.run] += 1

            self.parents = self.parent_selection()
            if not self.parents:  # Convergence
                break

            self.children = self.parent_crossover()

            self.children_mutate()

            self.hj.population = self.update_population()

    def parent_selection(self):
        # Stochastic Universal Sampling
        max_fitness = sum([particle.fitness for particle in self.hj.population])
        if max_fitness == 0:
            return False

        #fitness_proportionate = [particle.fitness / max_fitness for particle in self.hj.population]  # For maximisation

        # Fitness proportionate where smaller fitness is better
        fitness_proportionate = [((max_fitness - particle.fitness) / max_fitness) / (len(self.hj.population) - 1) for
                                 particle in self.hj.population]

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
            child = self.hj.crossover(self.hj.population[self.parents[0]].candidate,
                                      self.hj.population[self.parents[1]].candidate)
            children.append(child)

        return children

    def children_mutate(self):
        # Only mutate if parents too similar i.e. match genes retaining order significance
        # Situation occurs more frequently in combinatorial problems than continuous (represented by binary chromosomes)
        parent_gene_similarity_index = [i for i, j in zip(self.hj.population[self.parents[0]].candidate,
                                                          self.hj.population[self.parents[1]].candidate) if i == j]

        if len(parent_gene_similarity_index) > (len(self.hj.population[self.parents[0]].candidate) *
                                                self.hj.parent_gene_similarity_threshold):
            for i in range(self.hj.number_children):
                self.children[i] = self.hj.variator(self.children[i])

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

    def reset_inherited_population_attr(self):

        import struct
        for c in self.hj.population:
            if isinstance(c.candidate[0], float):
                cand = []
                for gene in c.candidate:
                    gene = format(struct.unpack('!I', struct.pack('!f', gene))[0], '032b')
                    t = [int(x) for x in gene]
                    cand.append(t)
                c.candidate = cand
