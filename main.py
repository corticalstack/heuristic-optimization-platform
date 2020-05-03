from random import Random
from utils.visualisation import Visualisation
from config.config import *
import os
import sys
from datetime import datetime
import copy
from problems.fssp import *
from optimizers.rnd import RND
from optimizers.shc import SHC
from optimizers.es import ES
from optimizers.de import DE
from optimizers.sa import SA
from optimizers.ga import GA
from optimizers.pso import PSO

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
        self.opt_runtime_stats = {'best_cf': 999999999, 'best_cp': [], 'lb_diff_pct': 0, 'ub_diff_pct': 0, 'avg_cts': 0,
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
                
        except KeyError:
            lg.msg(logging.INFO, 'Key error during optimization of problem benchmark instances. Terminating')

        lg.msg(logging.INFO, 'Flow shop scheduling problem completed')

    def optimize_problem_benchmark_instance(self, pid, iid, oid):
        lg.msg(logging.INFO, 'Optimizing problem benchmark with optimizer {}'.format(oid))
        
        # Get class for problem and instantiate 
        cls = globals()[pid]
        problem = cls(self.random, self.cfg, oid, iid)

        # Get class for optimizer and instantiate
        cls = globals()[oid]
        optimizer = cls(self.random, self.cfg, problem)

        # Execute optimizer configured number of times to sample problem results
        lg.msg(logging.INFO, 'Executing {} sample runs'.format(self.cfg.settings['gen']['runs_per_optimizer']))

        total_cts = 0  # Total computational time in seconds
        for i in range(self.cfg.settings['gen']['runs_per_optimizer']):

            opt_run_start_time = time.time()
            optimizer.before_start()
            run_best_cf, run_best_cp, run_ft = optimizer.optimize()  # Execute optimizer
            optimizer.on_completion()
            total_cts += time.time() - opt_run_start_time  # Aggregate computational time this run to total

            lg.msg(logging.INFO, 'Run {} best fitness is {} with permutation {}'.format(i, run_best_cf, run_best_cp))

            if run_best_cf < self.cfg.settings['opt'][oid]['best_cf']:
                self.cfg.settings['opt'][oid]['best_cf'] = run_best_cf
                self.cfg.settings['opt'][oid]['best_cp'] = run_best_cp

            # Log best fitness for this run to see trend over execution runs
            self.cfg.settings['opt'][oid]['ft'].append(run_best_cf)

            self.vis.fitness_trend(run_ft)  # Plot run-specific trend

        # Log optimizer average completion time seconds
        self.cfg.settings['opt'][oid]['avg_cts'] = total_cts / self.cfg.settings['gen']['runs_per_optimizer']

        # Execute problem-specific tasks upon optimization completion, for e.g. generate gantt chart of best schedule
        problem.on_completion(self.cfg, oid)

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

