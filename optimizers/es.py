from optimizers.optimizer import Optimizer
import logging
from utilities import logger as lg
import inspyred
from optimizers.inspyred_wrapper import InspyredWrapper


class ES(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

    def optimize(self):
        self.evolve()

    def evolve(self):
        es = inspyred.ec.ES(self.random)
        es.observer = InspyredWrapper.observer
        es.terminator = [inspyred.ec.terminators.evaluation_termination,
                         inspyred.ec.terminators.diversity_termination]

        final_pop = es.evolve(generator=InspyredWrapper.generator,
                              evaluator=InspyredWrapper.evaluator,
                              pop_size=self.hj.initial_pop_size,
                              maximize=False,
                              max_evaluations=self.hj.budget,
                              slf=self)

        final_pop.sort(reverse=True)
        self.hj.rbest.fitness = final_pop[0].fitness

        # Inspyred ES extends candidate with strategy elements, slice for actual solution cand. associated with fitness
        self.hj.rbest.candidate = final_pop[0].candidate[:self.hj.pid_cls.n]
        self.hj.rft = list(set(self.hj.rft))  # Remove duplicates
        self.hj.rft.sort(reverse=True)
