from optimizers.optimizer import Optimizer
import logging
from utilities import logger as lg
import inspyred
from optimizers.inspyred_wrapper import InspyredWrapper


class DEA(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)
        self.initial_candidate_size = 30
        lg.msg(logging.DEBUG, 'Population size to {}'.format(self.initial_candidate_size))

        self.pos_min = 0
        self.pos_max = 4

    def optimize(self):
        self.evolve()
        return self.gbest.fitness, self.gbest.candidate, self.fitness_trend

    def evolve(self):
        dea = inspyred.ec.DEA(self.random)
        dea.observer = InspyredWrapper.observer
        dea.terminator = inspyred.ec.terminators.evaluation_termination

        final_pop = dea.evolve(generator=InspyredWrapper.generator,
                               evaluator=InspyredWrapper.evaluator,
                               pop_size=self.initial_candidate_size,
                               maximize=False,
                               max_generations=self.budget,
                               slf=self,
                               problem=self.problem,
                               cfg=self.cfg)

        final_pop.sort(reverse=True)
        self.gbest.fitness = final_pop[0].fitness
        self.gbest.candidate = self.problem.candidate_spv_continuous_to_discrete(final_pop[0].candidate)

        self.fitness_trend = list(set(self.fitness_trend))  # Remove duplicates
        self.fitness_trend.sort(reverse=True)
