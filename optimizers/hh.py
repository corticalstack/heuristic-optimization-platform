from optimizers.hyper import Hyper
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import math
import numpy as np
from importlib import import_module
import collections
import copy

# class HHSelector(gym.Env):
#     """Hotter Colder
#     The goal of hotter colder is to guess closer to a randomly selected number
#
#     After each step the agent receives an observation of:
#     0 - No guess yet submitted (only after reset)
#     1 - Guess is lower than the target
#     2 - Guess is equal to the target
#     3 - Guess is higher than the target
#
#     The rewards is calculated as:
#     (min(action, self.number) + self.range) / (max(action, self.number) + self.range)
#
#     Ideally an agent will be able to recognise the 'scent' of a higher reward and
#     increase the rate in which is guesses in that direction until the reward reaches
#     its maximum
#     """
#     def __init__(self):
#         self.range = 1000  # +/- value the randomly select number can be between
#         self.bounds = 2000  # Action space bounds
#
#         self.action_space = spaces.Discrete(2)
#         self.observation_space = spaces.Discrete(5)
#
#         self.number = 0
#         self.guess_count = 0
#         self.guess_max = 200
#         self.observation = 0
#
#         self.seed()
#         self.reset()
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def step(self, action):
#         assert self.action_space.contains(action)
#
#         if action < self.number:
#             self.observation = 1
#
#         elif action == self.number:
#             self.observation = 2
#
#         elif action > self.number:
#             self.observation = 3
#
#         # JP could set reward according to whether fitness value beats best
#
#         reward = ((min(action, self.number) + self.bounds) / (max(action, self.number) + self.bounds)) ** 2
#
#         self.guess_count += 1
#         done = self.guess_count >= self.guess_max
#
#         return self.observation, reward, done, {"number": self.number, "guesses": self.guess_count}
#
#     def reset(self):
#         self.number = self.np_random.uniform(-self.range, self.range)
#         self.guess_count = 0
#         self.observation = 0
#         return self.observation


class HH(Hyper):
    def __init__(self, **kwargs):
        Hyper.__init__(self, **kwargs)

        # Optimizer specific
        self.epsilon = 0.3

        self.low_level_heuristics = collections.OrderedDict()
        self.import_low_level_heuristics()
        self.llh_num = len(self.cfg.settings['opt'][self.problem.oid]['low_level_selection_pool'])
        self.llc_initial_budget = 300
        self.llc_budget = 10000
        self.llh_fitness = []
        self.llh_perms = []

    def optimize(self):
        self.hyper()

    def hyper(self):
        self.reset_llh_archive()
        self.set_llh_samples()

        while self.budget > 0:
            bcf, bcp, llh = self.best_candidate_from_pool()

            if self.random.random() < self.epsilon:
                llh = self.random.randint(0, self.llh_num - 1)

            gbest = self.set_gbest(bcf, bcp)
            pop = self.set_pop()
            cls = self.low_level_heuristics[llh]['cls']
            run_best, run_ft, run_budget = cls.run(budget=self.llc_budget, gbest=gbest, pop=pop)
            print('Run best is ', run_best.fitness, ' with perm ', run_best.perm, ' by ', self.low_level_heuristics[llh]['llc'])
            self.budget += run_budget
            self.budget -= self.llc_budget

            if run_best.fitness < bcf:
                self.llh_fitness[llh].insert(0, run_best.fitness)  # Insert at start
                self.llh_perms[llh].insert(0, run_best.perm)

    def import_low_level_heuristics(self):
        try:
            for hci, hc in enumerate(self.cfg.settings['opt'][self.problem.oid]['low_level_selection_pool']):
                my_module = import_module('optimizers.' + hc.lower())
                cls = getattr(my_module, hc)(random=self.random, cfg=self.cfg, problem=self.problem)
                self.low_level_heuristics[hci] = {'llc': hc, 'cls': cls}
        except (ModuleNotFoundError, AttributeError) as e:
            lg.msg(logging.INFO, 'HH error {} importing low-level selection pool'.format(e))

    def best_candidate_from_pool(self):
        best = min((min((v, c) for c, v in enumerate(row)), r) for r, row in enumerate(self.llh_fitness))
        bcf = self.llh_fitness[best[1]][best[0][1]]
        bcp = self.llh_perms[best[1]][best[0][1]]
        bcllh = best[1]
        print('Best fitness = ', bcf, ' with permutation ', bcp)
        return bcf, bcp, bcllh

    def reset_llh_archive(self):
        self.llh_fitness = [[] for i in range(self.llh_num)]
        self.llh_perms = [[] for i in range(self.llh_num)]
        #self.llh_fitness = [[]] * self.llh_num
        #self.llh_perms = [[]] * self.llh_num

    def set_llh_samples(self):
        # Initialise starting samples
        for k, v in self.low_level_heuristics.items():
            for i in range(self.cfg.settings['opt'][self.problem.oid]['low_level_sample_runs']):
                run_best, run_ft, run_budget = v['cls'].run(budget=self.llc_initial_budget)
                self.budget += run_budget
                self.budget -= self.llc_initial_budget
                self.llh_fitness[k].append(run_best.fitness)  # Insert at start
                self.llh_perms[k].append(run_best.perm)

    def set_gbest(self, bcf, bcp):
        gbest = Particle()
        gbest.fitness = bcf
        gbest.perm = bcp
        return gbest

    def set_pop(self):
        # JP need to fix the list zip below
        candidates = list(zip(*self.llh_fitness, *self.llh_perms))  # Unpack into single list
        candidates.sort()
        population = []
        for fitness, perm in candidates[:1]:
            candidate = Particle()
            candidate.fitness = fitness
            candidate.perm = perm
            population.append(candidate)
        return population

