from optimizers.optimizer import Optimizer
import logging
from utilities import logger as lg
import inspyred
from optimizers.inspyred_wrapper import InspyredWrapper


class ES(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)
        self.initial_candidate_size = 30
        lg.msg(logging.DEBUG, 'Population size to {}'.format(self.initial_candidate_size))

        self.pos_min = 0
        self.pos_max = 4

    def optimize(self):
        self.evolve()
        return self.gbest.fitness, self.gbest.perm, self.fitness_trend

    def evolve(self):
        es = inspyred.ec.ES(self.random)
        es.observer = InspyredWrapper.observer
        es.terminator = [inspyred.ec.terminators.evaluation_termination,
                         inspyred.ec.terminators.diversity_termination]

        final_pop = es.evolve(generator=InspyredWrapper.generator,
                              evaluator=InspyredWrapper.evaluator,
                              pop_size=self.initial_candidate_size,
                              maximize=False,
                              max_evaluations=self.budget,
                              slf=self,
                              problem=self.problem,
                              cfg=self.cfg)

        final_pop.sort(reverse=True)
        self.gbest.fitness = final_pop[0].fitness

        # Inspyred ES extends candidate with strategy elements, slice for actual solution perm associated with fitness
        self.gbest.perm = self.problem.perm_spv_continuous_to_discrete(final_pop[0].candidate[:self.problem.n])
        self.fitness_trend = list(set(self.fitness_trend))  # Remove duplicates
        self.fitness_trend.sort(reverse=True)
