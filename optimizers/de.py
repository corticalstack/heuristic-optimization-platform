from optimizers.optimizer import Optimizer
import logging
from utils import logger as lg
import inspyred
from optimizers.inspyred_wrapper import InspyredWrapper


class DE(Optimizer):
    def __init__(self, random, cfg, prb):
        Optimizer.__init__(self, random, cfg, prb)
        self.initial_candidate_size = 30
        lg.msg(logging.DEBUG, 'Population size to {}'.format(self.initial_candidate_size))

        self.pos_min = 0
        self.pos_max = 4

    def optimize(self):
        self.prb.budget['remaining'] = self.prb.budget['total']
        self.evolve()
        return self.global_best.fitness, self.global_best.perm, self.fitness_trend

    def evolve(self):
        dea = inspyred.ec.DEA(self.random)
        dea.observer = InspyredWrapper.observer
        dea.terminator = inspyred.ec.terminators.evaluation_termination

        final_pop = dea.evolve(generator=InspyredWrapper.generator,
                               evaluator=InspyredWrapper.evaluator,
                               pop_size=self.initial_candidate_size,
                               maximize=False,
                               max_generations=self.prb.budget['total'],
                               slf=self,
                               prb=self.prb,
                               cfg=self.cfg)

        final_pop.sort(reverse=True)
        self.global_best.fitness = final_pop[0].fitness
        self.global_best.perm = self.prb.perm_spv_continuous_to_discrete(final_pop[0].candidate)

        self.fitness_trend = list(set(self.fitness_trend))  # Remove duplicates
        self.fitness_trend.sort(reverse=True)
