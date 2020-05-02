from optimizers.optimizer import Optimizer
from optimizers.particle import Particle
import logging
from utils import logger as lg
import math
import numpy as np
import inspyred


class ACO(Optimizer):
    def __init__(self, random, cfg, prb):
        Optimizer.__init__(self, random, cfg, prb)
        self.initial_candidate_size = 10
        lg.msg(logging.DEBUG, 'Swarm size to {}'.format(self.initial_candidate_size))

        self.current_generation = 1
        self.candidate_id = 0
        self.candidate_fitness = []
        self.population = []

        self.min = 0
        self.max = 4
        self.velocity_clip = (-4, 4)

    def optimize(self):
        self.prb.budget['remaining'] = self.prb.budget['total']
        self.colony()
        return self.global_best.fitness, self.global_best.perm, self.fitness_trend

    @staticmethod
    def generate_solution(random, args):
        prb = args['prb']
        cfg = args['cfg']

        perm = getattr(prb, 'generator_' + cfg.settings['opt']['ACO']['generator'])()
        return perm

    @staticmethod
    def transform_cont_perm(particle):
        # Get smallest position value
        spv = sorted(range(len(particle.perm_cont)), key=lambda i: particle.perm_cont[i], reverse=False)
        return spv

    @staticmethod
    def evaluator(candidates, args):
        # JP check the budget is reduced here and how can I interupt Inspyred
        fitness = []
        for c in candidates:
            if isinstance(c[0], float):
                c = sorted(range(len(c)), key=lambda i: c[i], reverse=False)
            f = args['prb'].evaluator(c)
            fitness.append(f)
        return fitness



    def colony(self):
        components = list(range(0, 20))

        # DEA
        # ac = inspyred.ec.DEA(prng)
        # ac.terminator = inspyred.ec.terminators.evaluation_termination
        # final_pop = ac.evolve(generator=self.generate_solution,
        #                       evaluator=self.evaluator,
        #                       pop_size=100,
        #                       prb=self.prb,
        #                       cfg=self.cfg,
        #                       maximize=False,
        #                       max_generations=20000)

        # ES
        ac = inspyred.ec.ES(self.random)
        ac.terminator = [inspyred.ec.terminators.evaluation_termination,
                         inspyred.ec.terminators.diversity_termination]
        final_pop = ac.evolve(generator=self.generate_solution,
                              evaluator=self.evaluator,
                              pop_size=100,
                              prb=self.prb,
                              cfg=self.cfg,
                              maximize=False,
                              max_evaluations=20000)

        # Sort and print the best individual, who will be at index 0.
        final_pop.sort(reverse=True)
        print(final_pop[0])

    @staticmethod
    def transform_continuous_permutation(candidate):
        # Get smallest position value
        spv = sorted(range(len(candidate)), key=lambda i: candidate[i], reverse=False)
        return spv

