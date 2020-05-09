from random import Random
from utilities.visualisation import Visualisation
from config.config import *
import os
import sys
from datetime import datetime
import copy
from problems.fssp import *
from optimizers.rnd import RND
from optimizers.shc import SHC
from optimizers.es import ES
from optimizers.dea import DEA
from optimizers.sa import SA
from optimizers.ga import GA
from optimizers.pso import PSO
from optimizers.hh import HH

import time

script_name = os.path.basename(sys.argv[0]).split('.')


class HeuristicOptimizerPlatform:
    """
    Heuristic Optimizer Platform
    """
    def __init__(self):
        lg.msg(logging.INFO, 'Starting Heuristic Optimizer Platform (HOP)')

        self.random = Random()
        self.random.seed(42)

        self.cfg = Config()
        self.vis = Visualisation()

        # Add runtime stats template to each optimizer
        self.opt_runtime_stats = {'bcf': 999999999, 'bcp': [], 'lb_diff_pct': 0, 'ub_diff_pct': 0, 'avg_cts': 0,
                                  'ft': []}
        self.add_opt_runtime_stats()

        try:
            # Platform takes each optimization problem in turn
            for pid in self.cfg.settings['prb']:
                if not self.cfg.settings['prb'][pid]['enabled']:
                    continue
                lg.msg(logging.INFO, 'Processing {}'.format(self.cfg.settings['prb'][pid]['description']))

                # For the given problem, optimize each of the associated benchmark problems
                for iid in self.cfg.settings['ben'][pid]['instances']:
                    if not self.cfg.settings['ben'][pid]['instances'][iid]['enabled']:
                        continue
                    lg.msg(logging.INFO, 'Optimizing {} benchmark problem instance {}'.format(
                        self.cfg.settings['ben'][pid]['type'], iid))

                    # Optimize the given benchmark problem with enabled optimizers
                    for oid in self.cfg.settings['opt']:
                        if not self.cfg.settings['opt'][oid]['enabled']:
                            continue

                        self.optimize_problem_benchmark_instance(pid, iid, oid)

                # Summarise results as optimization has completed
                self.summary()
                
        except KeyError as e:
            lg.msg(logging.INFO, 'Key error {} during optimization. Terminating'.format(e))

        lg.msg(logging.INFO, 'Flow shop scheduling problem completed')

    def optimize_problem_benchmark_instance(self, pid, iid, oid):
        lg.msg(logging.INFO, 'Optimizing problem benchmark with optimizer {}'.format(oid))

        problem, optimizer = self.make_components(pid=pid, iid=iid, oid=oid)

        # Execute optimizer configured number of times to sample problem results
        lg.msg(logging.INFO, 'Executing {} sample runs'.format(self.cfg.settings['gen']['runs_per_optimizer']))

        total_cts = 0  # Total computational time in seconds
        for i in range(self.cfg.settings['gen']['runs_per_optimizer']):

            opt_run_start_time = time.time()

            run_best, run_ft, _ = optimizer.run(budget=problem.budget['total'])  # Execute optimizer

            total_cts += time.time() - opt_run_start_time  # Aggregate computational time this run to total

            lg.msg(logging.INFO, 'Run {} best fitness is {} with permutation {}'.format(i, run_best.fitness, run_best.candidate))
            self.log_optimizer_fitness(oid=oid, bcf=run_best.fitness, bcp=run_best.candidate)

            self.vis.fitness_trend(run_ft)  # Plot run-specific trend

        # Log optimizer average completion time seconds
        self.cfg.settings['opt'][oid]['avg_cts'] = total_cts / self.cfg.settings['gen']['runs_per_optimizer']

        # Execute problem-specific tasks upon optimization completion, for e.g. generate gantt chart of best schedule
        problem.post_processing(oid=oid)

    def load_components(self):
        pass
        # load components

    # make components
    def make_components(self, **kwargs):
        # Get class for problem and instantiate
        cls = globals()[kwargs['pid']]
        problem = cls(random=self.random, cfg=self.cfg, oid=kwargs['oid'], iid=kwargs['iid'])

        # Get class for optimizer and instantiate
        cls = globals()[self.cfg.settings['opt'][kwargs['oid']]['optimizer']]
        optimizer = cls(random=self.random, cfg=self.cfg, problem=problem)

        return problem, optimizer

    def log_optimizer_fitness(self, **kwargs):
        if kwargs['bcf'] < self.cfg.settings['opt'][kwargs['oid']]['bcf']:
            self.cfg.settings['opt'][kwargs['oid']]['bcf'] = kwargs['bcf']
            self.cfg.settings['opt'][kwargs['oid']]['bcp'] = kwargs['bcp']

        # Log best fitness for this run to see trend over execution runs
        self.cfg.settings['opt'][kwargs['oid']]['ft'].append(kwargs['bcf'])

    def add_opt_runtime_stats(self):
        for oid in self.cfg.settings['opt']:
            if not self.cfg.settings['opt'][oid]['enabled']:
                continue
            stats_template = copy.deepcopy(self.opt_runtime_stats)
            self.cfg.settings['opt'][oid].update(stats_template)

    def summary(self):
        self.vis.fitness_trend_all_optimizers(self.cfg.settings['opt'])
        Stats.summary(self.cfg.settings['opt'])


if __name__ == "__main__":
    log_filename = str('hop_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt')

    logging.basicConfig(filename='logs/' + log_filename, level=logging.INFO,
                        format='[%(asctime)s] [%(levelname)8s] %(message)s')

    # Disable matplotlib font manager logger
    logging.getLogger('matplotlib.font_manager').disabled = True

    hop = HeuristicOptimizerPlatform()

