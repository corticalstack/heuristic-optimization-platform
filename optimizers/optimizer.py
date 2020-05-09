from optimizers.particle import Particle
import logging
from utilities import logger as lg
import copy


class Optimizer:
    def __init__(self, **kwargs):
        # Persist current configuration and problem
        self.random = kwargs['random']
        self.cfg = kwargs['cfg']
        self.problem = kwargs['problem']

        self.logger = logging.getLogger()
        self.initial_candidate_size = 1
        self.gbest = None
        self.population = []
        self.fitness_trend = []
        self.budget = 0

        # Initial sample may be used to determine search starting point
        try:
            if 'initial_sample' in self.cfg.settings['opt'][self.__class__.__name__]:
                if self.cfg.settings['opt'][self.__class__.__name__]['initial_sample']:
                    self.problem.initial_sample = self.problem.generate_initial_sample(self.__class__.__name__)
        except KeyError:
            pass

    def run(self, **kwargs):
        self.budget = kwargs['budget']
        self.pre_processing(kwargs)
        self.optimize()
        self.post_processing()

        gbest = copy.deepcopy(self.gbest)
        return gbest, self.fitness_trend, self.budget

    def n_swap(self, candidate):
        # This does a local search by swapping two random jobs
        new_candidate = candidate.copy()
        idx = self.random.sample(range(0, len(new_candidate)), 2)

        # Pair swap
        new_candidate[idx[0]], new_candidate[idx[1]] = new_candidate[idx[1]], new_candidate[idx[0]]
        return new_candidate

    def n_insert(self):
        pass

    def n_exchange(self):
        pass

    def n_shift(self):
        pass

    def n_sbox(self):
        pass

    def pre_processing(self, kwargs):
        self.fitness_trend = []

        # Set global best single particle if passed
        if 'gbest' in kwargs:
            self.gbest = kwargs['gbest']
        else:
            self.gbest = Particle()

        # Set population of particles if passed
        if 'pop' in kwargs:
            self.population = kwargs['pop']
        else:
            self.population = []

    def post_processing(self):
        lg.msg(logging.DEBUG, 'Computational budget remaining is {}'.format(self.budget))

