from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
from importlib import import_module
import copy
import collections


class Hyper(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)
        self.low_level_heuristics = collections.OrderedDict()
        self.llh_total = len(self.hj.low_level_selection_pool)
        self.llh_fitness = []
        self.llh_candidates = []
        self.llh_exec = []
        self.jobs = []

    def pre_processing(self, **kwargs):
        self.jobs = kwargs['jobs']
        self.import_low_level_heuristics()
        self.llh_fitness = [[] for i in range(self.llh_total)]
        self.llh_candidates = [[] for i in range(self.llh_total)]
        self.llh_exec = [[] for i in range(self.llh_total)]

    def post_processing(self):
        Optimizer.post_processing(self)
        print('Finished with best of ', self.hj.rbest.fitness)
        for k, v in self.low_level_heuristics.items():
            print('Llh {} executed {} times and with aggregated improvements of {}'.format(v.oid, v.oid_run_count, v.oid_aggr_imp))

    def best_candidate_from_pool(self):
        best = min((min((v, c) for c, v in enumerate(row)), r) for r, row in enumerate(self.llh_fitness))
        bcf = self.llh_fitness[best[1]][best[0][1]]
        bc = self.llh_candidates[best[1]][best[0][1]]
        bcllh = best[1]
        #print('{} set best fitness {} with candidate {}'.format(self.low_level_heuristics[bcllh].oid, bcf, bc))
        return bcf, bc, bcllh

    def set_rbest(self, bcf, bc):
        self.hj.rbest.fitness = bcf
        self.hj.rbest.candidate = bc

    def set_llh_samples(self):
        # Initialise starting samples
        for k, v in self.low_level_heuristics.items():
            for i in range(3):
                v.budget = self.hj.llh_sample_budget
                v.oid_cls.run()
                self.hj.budget -= self.hj.llh_sample_budget
                self.hj.budget += v.budget  # Credit any early termination or debit any budget overrun
                self.llh_fitness[k].append(v.rbest.fitness)  # Insert at start
                self.llh_candidates[k].append(v.rbest.candidate)

    def add_samples_to_trend(self):
        self.hj.rft = [y for x in self.llh_fitness for y in x]
        self.hj.rft.sort(reverse=True)

    def set_pop(self):
        candidates = list(zip([y for x in self.llh_fitness for y in x], [y for x in self.llh_candidates for y in x]))
        candidates.sort()
        population = []

        for fitness, candidate in candidates[:1]:
            c = Particle()
            c.fitness = fitness
            c.candidate = candidate
            population.append(c)
        return population

    def import_low_level_heuristics(self):
        for hci, hc in enumerate(self.hj.low_level_selection_pool):
            llh = [x for x in self.jobs if x.pid == self.hj.pid and x.oid == hc]
            self.low_level_heuristics[hci] = llh[0]
