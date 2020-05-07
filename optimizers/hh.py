from optimizers.hyper import Hyper
from optimizers.particle import Particle
import logging
from utilities import logger as lg
import math
import numpy as np
from importlib import import_module
import collections


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
        self.low_level_heuristics = collections.OrderedDict()
        self.llc_initial_budget = 400
        self.llc_budget = 5000

        self.trends = []
        self.permutations = []

    def optimize(self):
        self.import_low_level_heuristics()
        self.hyper()
        return self.gbest.fitness, self.gbest.perm, self.fitness_trend

    def import_low_level_heuristics(self):
        try:
            for hci, hc in enumerate(self.cfg.settings['opt'][self.problem.oid]['low_level_selection_pool']):
                my_module = import_module('optimizers.' + hc.lower())
                cls = getattr(my_module, hc)(random=self.random, cfg=self.cfg, problem=self.problem)
                self.low_level_heuristics[hci] = {'llc': hc, 'cls': cls}
                self.trends.append([])
                self.permutations.append([])
        except (ModuleNotFoundError, AttributeError) as e:
            lg.msg(logging.INFO, 'HH error {} importing low-level selection pool'.format(e))

    # n state environment - relating to the number of low level heuristics
    # 2 possible actions relating to each state - call or not call

    def get_llc(self):
        pass

    def hyper(self):

        # Initialise starting samples
        for k, v in self.low_level_heuristics.items():
            run_bcf, run_bcp, run_ft = v['cls'].run(budget=self.llc_initial_budget)
            self.budget -= self.llc_initial_budget
            self.trends[k].insert(0, run_bcf)  # Insert at start
            self.permutations[k].insert(0, run_bcp)

        eps = 0.5
        while self.budget > 0:
            gbest = min((min((v, c) for c, v in enumerate(row)), r) for r, row in enumerate(self.trends))
            print('Row ', gbest[1], 'Column ', gbest[0][1])
            print('Best fitness = ', self.trends[gbest[1]][gbest[0][1]], ' with permutation ',
                  self.permutations[gbest[1]][gbest[0][1]])
            llc = gbest[1]
            if np.random.random() < eps:
                llc = self.random.randint(0, len(self.cfg.settings['opt'][self.problem.oid]['low_level_selection_pool']) - 1)

            # JP - Need to pass best permutation for the heuristic to start with
            run_bcf, run_bcp, run_ft = self.low_level_heuristics[llc]['cls'].run(budget=self.llc_budget)
            self.budget -= self.llc_budget
            self.trends[llc].insert(0, run_bcf)  # Insert at start
            self.permutations[llc].insert(0, run_bcp)
            pass




