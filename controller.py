from random import Random
from utilities.visualisation import Visualisation
from config.config import *
import os
import sys
from itertools import product
from datetime import datetime
import copy

from problems.fssp import *
from problems.rastrigin import *
from optimizers.rnd import RND
from optimizers.shc import SHC
from optimizers.es import ES
from optimizers.dea import DEA
from optimizers.sa import SA
from optimizers.ga import GA
from optimizers.pso import PSO
from optimizers.hh import HH
from hopjob import HopJob
import time

script_name = os.path.basename(sys.argv[0]).split('.')

import yaml


class Controller:
    """
    Controller
    """
    def __init__(self):
        lg.msg(logging.INFO, 'Initialising controller')
        self.random = Random()
        self.random.seed(42)
        self.vis = Visualisation()
        self.settings = self.get_config()
        self.jobs = self.set_jobs()

        # Add runtime stats template to each optimizer
        #self.opt_runtime_stats = {'bcf': 999999999, 'bcp': [], 'lb_diff_pct': 0, 'ub_diff_pct': 0, 'avg_cts': 0,
        #                          'ft': []}
        #self.add_opt_runtime_stats()

    @staticmethod
    def get_config():
        _settings = {}
        _config_files = {
            'gen': 'config/general.yaml',
            'prb': 'config/problems.yaml',
            'opt': 'config/optimizers.yaml'
        }

        for k, f, in _config_files.items():
            with open(f, 'r') as stream:
                try:
                    _settings[k] = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    print(e)
        return _settings

    def set_jobs(self):
        jobs = []
        prob = self.get_enabled_problems()
        opt = self.get_enabled_optimizers()
        prob_opt = list(product(prob, opt))

        for (pid, oid) in prob_opt:
            benchmarks = []
            if 'benchmarks' not in self.settings['prb'][pid]:
                benchmarks.append('n/a')
            else:
                for bid in self.settings['prb'][pid]['benchmarks']:
                    if not self.settings['prb'][pid]['benchmarks'][bid]['enabled']:
                        continue
                    benchmarks.append(bid)
            for bid in benchmarks:
                jobs.append(self.add_job(pid, oid, bid))
        return jobs

    def add_job(self, *args):
        job = HopJob()
        job.pid, job.oid, job.bid = args
        job.comp_budget_base = self.settings['gen']['comp_budget_base']
        job.runs_per_optimizer = self.settings['gen']['runs_per_optimizer']

        if 'initial_sample' in self.settings['opt'][job.oid]:
            job.initial_sample = self.settings['opt'][job.oid]['initial_sample']

        cls = globals()[job.pid]
        job.pid_cls = cls(random=self.random, hopjob=job)  # Instantiate problem
        job.budget = job.pid_cls.n * job.comp_budget_base
        cls = globals()[job.oid]
        job.oid_cls = cls(random=self.random, hopjob=job)  # Instantiate optimizer

        job.pid_lb = self.settings['prb'][job.pid]['lb']
        if self.settings['prb'][job.pid]['ub'] == 'nmax':
            job.pid_ub = job.pid_cls.n
        else:
            job.pid_ub = self.settings['prb'][job.pid]['ub']

        job.oid_lb = self.settings['opt'][job.oid]['lb']
        job.oid_ub = self.settings['opt'][job.oid]['ub']

        # Optimizer configured generator overrides higher level problem generator e.g. PSO works on continuous values
        if 'generator' in self.settings['opt'][job.oid]:
            job.generator = getattr(job.pid_cls, 'generator_' + self.settings['opt'][job.oid]['generator'])
        else:
            job.generator = getattr(job.pid_cls, 'generator_' + self.settings['prb'][job.pid]['generator'])

        return job

    def get_enabled_problems(self):
        problems = []
        for pid in self.settings['prb']:
            if not self.settings['prb'][pid]['enabled']:
                continue
            problems.append(pid)
        return problems

    def get_enabled_optimizers(self):
        optimizers = []
        for oid in self.settings['opt']:
            if not self.settings['opt'][oid]['enabled']:
                continue
            optimizers.append(oid)
        return optimizers

    def execute_jobs(self):
        for j in self.jobs:
            j.start_time = time.time()

            # Execute optimizer configured number of times to sample problem results
            lg.msg(logging.INFO, 'Executing {} sample runs'.format(j.runs_per_optimizer))

            for r in range(j.runs_per_optimizer):
                exec_start_time = time.time()
                j.oid_cls.run()
                j.end_time = time.time()
                j.comp_time_s += time.time() - exec_start_time

                lg.msg(logging.INFO, 'Run {} best fitness is {} with permutation {}'.format(r, j.gbest.fitness, j.gbest.candidate))
                self.vis.fitness_trend(j.fitness_trend)  # Plot run-specific trend
            j.end_time = time.time()

    def optimize_problem(self, **kwargs):
        problem, optimizer = self.make_components(**kwargs)

        # Execute optimizer configured number of times to sample problem results
        lg.msg(logging.INFO, 'Executing {} sample runs'.format(self.settings['gen']['runs_per_optimizer']))

        total_cts = 0  # Total computational time in seconds
        for i in range(self.settings['gen']['runs_per_optimizer']):

            opt_run_start_time = time.time()

            run_best, run_ft, _ = optimizer.run(budget=problem.budget['total'])  # Execute optimizer

            total_cts += time.time() - opt_run_start_time  # Aggregate computational time this run to total

            lg.msg(logging.INFO, 'Run {} best fitness is {} with permutation {}'.format(i, run_best.fitness, run_best.candidate))
            self.log_optimizer_fitness(oid=kwargs['oid'], bcf=run_best.fitness, bcp=run_best.candidate)

            self.vis.fitness_trend(run_ft)  # Plot run-specific trend

        # Log optimizer average completion time seconds
        self.settings['opt'][kwargs['oid']]['avg_cts'] = total_cts / self.settings['gen']['runs_per_optimizer']

        # Execute problem-specific tasks upon optimization completion, for e.g. generate gantt chart of best schedule
        problem.post_processing(oid=kwargs['oid'])

    def load_components(self):
        pass
        # load components

    def log_optimizer_fitness(self, **kwargs):
        if kwargs['bcf'] < self.settings['opt'][kwargs['oid']]['bcf']:
            self.settings['opt'][kwargs['oid']]['bcf'] = kwargs['bcf']
            self.settings['opt'][kwargs['oid']]['bcp'] = kwargs['bcp']

        # Log best fitness for this run to see trend over execution runs
        self.settings['opt'][kwargs['oid']]['ft'].append(kwargs['bcf'])

    def add_opt_runtime_stats(self):
        for oid in self.settings['opt']:
            if not self.settings['opt'][oid]['enabled']:
                continue
            stats_template = copy.deepcopy(self.opt_runtime_stats)
            self.settings['opt'][oid].update(stats_template)

    def summary(self):
        self.vis.fitness_trend_all_optimizers(self.settings['opt'])
        Stats.summary(self.settings['opt'])
