from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utilities import logger as lg
from importlib import import_module
import collections


class Hyper(Optimizer):
    def __init__(self, **kwargs):
        Optimizer.__init__(self, **kwargs)

        self.low_level_heuristics = collections.OrderedDict()
        self.import_low_level_heuristics()
        self.llh_total = len(self.cfg.settings['opt'][self.problem.oid]['low_level_selection_pool'])
        self.llh_initial_budget = 300
        self.llh_budget = 1600
        self.llh_fitness = []
        self.llh_candidates = []
        self.llh_exec = []

    def pre_processing(self, kwargs):
        Optimizer.pre_processing(self, kwargs)
        for k, v in self.low_level_heuristics.items():
            v['run_count'] = 0
            v['aggr_imp'] = 0

        self.llh_fitness = [[] for i in range(self.llh_total)]
        self.llh_candidates = [[] for i in range(self.llh_total)]
        self.llh_exec = [[] for i in range(self.llh_total)]

    def post_processing(self):
        Optimizer.post_processing(self)
        print('Finished with best of ', self.gbest.fitness)
        for k, v in self.low_level_heuristics.items():
            print('Llh {} executed {} times and with aggregated improvements of {}'.format(v['llh'], v['run_count'], v['aggr_imp']))

    def best_candidate_from_pool(self):
        best = min((min((v, c) for c, v in enumerate(row)), r) for r, row in enumerate(self.llh_fitness))
        bcf = self.llh_fitness[best[1]][best[0][1]]
        bc = self.llh_candidates[best[1]][best[0][1]]
        bcllh = best[1]
        print('{} set best fitness {} with candidate {}'.format(self.low_level_heuristics[bcllh]['llh'], bcf, bc))
        return bcf, bc, bcllh

    def set_gbest(self, bcf, bcp):
        self.gbest.fitness = bcf
        self.gbest.candidate = bcp

    def set_llh_samples(self):
        # Initialise starting samples
        for k, v in self.low_level_heuristics.items():
            for i in range(self.cfg.settings['opt'][self.problem.oid]['low_level_sample_runs']):
                run_best, run_ft, run_budget = v['cls'].run(budget=self.llh_initial_budget)
                self.budget += run_budget
                self.budget -= self.llh_initial_budget
                self.llh_fitness[k].append(run_best.fitness)  # Insert at start
                self.llh_candidates[k].append(run_best.candidate)

    def add_samples_to_trend(self):
        self.fitness_trend = [y for x in self.llh_fitness for y in x]
        self.fitness_trend.sort(reverse=True)

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
        try:
            for hci, hc in enumerate(self.cfg.settings['opt'][self.problem.oid]['low_level_selection_pool']):
                my_module = import_module('optimizers.' + hc.lower())
                cls = getattr(my_module, hc)(random=self.random, cfg=self.cfg, problem=self.problem)
                self.low_level_heuristics[hci] = {'llh': hc, 'cls': cls, 'run_count': 0, 'aggr_imp': 0}
        except (ModuleNotFoundError, AttributeError) as e:
            lg.msg(logging.INFO, 'HH error {} importing low-level selection pool'.format(e))
