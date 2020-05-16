from optimizers.optimizer import Optimizer
import inspyred
from optimizers.inspyred_wrapper import InspyredWrapper


class DEA(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

    def optimize(self):
        self.evolve()

    def evolve(self):
        dea = inspyred.ec.DEA(self.random)
        dea.observer = InspyredWrapper.observer
        dea.terminator = inspyred.ec.terminators.evaluation_termination

        final_pop = dea.evolve(generator=InspyredWrapper.generator,
                               evaluator=InspyredWrapper.evaluator,
                               pop_size=self.hj.initial_pop_size,
                               maximize=False,
                               max_evaluations=self.hj.budget,
                               slf=self)

        final_pop.sort(reverse=True)
        self.hj.rbest.fitness = final_pop[0].fitness
        self.hj.rbest.candidate = final_pop[0].candidate

        self.hj.rft = list(set(self.hj.rft))  # Remove duplicates
        self.hj.rft.sort(reverse=True)
