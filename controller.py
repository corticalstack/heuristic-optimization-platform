from random import Random
from utilities.visualisation import Visualisation
from optimizers.particle import Particle
from config.config import *
import os
import sys
from itertools import product
from datetime import datetime
import copy

from problems.fssp import *
from problems.rastrigin import *
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
        self.problems_optimizers = []
        self.jobs = self.set_jobs()

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
        self.problems_optimizers = list(product(prob, opt))
        self.problems_optimizers.sort()

        for (pid, oid) in self.problems_optimizers:
            benchmarks = []
            if 'benchmarks' not in self.settings['prb'][pid]:
                benchmarks.append('n/a')
            else:
                for bid in self.settings['prb'][pid]['benchmarks']:
                    if not self.settings['prb'][pid]['benchmarks'][bid]['enabled']:
                        continue
                    benchmarks.append(bid)
            for bid in benchmarks:
                jobs.append(self.create_job_spec(pid, oid, bid))
        return jobs

    def create_job_spec(self, *args):
        job = HopJob()
        job.pid, job.oid, job.bid = args
        job.type = self.settings['prb'][job.pid]['type']
        job.comp_budget_base = self.settings['gen']['comp_budget_base']
        job.runs_per_optimizer = self.settings['gen']['runs_per_optimizer']
        job.bit_computing = self.settings['gen']['bit_computing']

        if 'initial_sample' in self.settings['opt'][job.oid]:
            job.initial_sample = self.settings['opt'][job.oid]['initial_sample']

        cls = globals()[job.pid]
        job.pid_cls = cls(random=self.random, hopjob=job)  # Instantiate problem
        job.budget = job.pid_cls.n * job.comp_budget_base
        if job.type == 'combinatorial':
            job.initial_candidate_size = job.pid_cls.n * 2
        else:
            job.initial_candidate_size = 5

        if 'number_parents' in self.settings['opt'][job.oid]:
            job.number_parents = self.settings['opt'][job.oid]['number_parents']

        if 'number_children' in self.settings['opt'][job.oid]:
            job.number_children = self.settings['opt'][job.oid]['number_children']

        job.pid_lb = self.settings['prb'][job.pid]['lb']
        if self.settings['prb'][job.pid]['ub'] == 'nmax':
            job.pid_ub = job.pid_cls.n
        else:
            job.pid_ub = self.settings['prb'][job.pid]['ub']

        job.oid_lb = self.settings['opt'][job.oid]['lb']
        job.oid_ub = self.settings['opt'][job.oid]['ub']
        cls = globals()[self.settings['opt'][job.oid]['optimizer']]
        job.oid_cls = cls(random=self.random, hopjob=job)  # Instantiate optimizer

        # Optimizer configured generator overrides higher level problem generator e.g. PSO works on continuous values
        if 'generator' in self.settings['opt'][job.oid] and job.type == 'continuous':
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
                self.pre_processing(j)
                j.oid_cls.run()
                j.end_time = time.time()
                j.total_comp_time_s += time.time() - exec_start_time

                if isinstance(j.rbest.candidate[0], float) and j.type == 'combinatorial':
                    j.rbest.candidate = j.pid_cls.candidate_spv_continuous_to_discrete(j.rbest.candidate)

                lg.msg(logging.INFO, 'Run {} best fitness is {} with candidate {}'.format(r, "{:.10f}".format(j.rbest.fitness), j.rbest.candidate))
                self.log_optimizer_fitness(j)

                self.vis.fitness_trend(j.rft)  # Plot run-specific trend

            j.end_time = time.time()
            j.avg_comp_time_s = j.total_comp_time_s / j.runs_per_optimizer

            # Execute problem-specific tasks upon optimization completion e.g. generate Gantt chart of best schedule
            j.pid_cls.post_processing()

        self.summary()

    def pre_processing(self, j):
        j.budget = j.pid_cls.n * j.comp_budget_base
        j.rft = []

        # Set global best single particle if passed
        # if 'gbest' in kwargs:
        #    self.gbest = kwargs['gbest']
        # else:
        j.rbest = Particle()

        # Set population of particles if passed
        # if 'pop' in kwargs:
        #    self.population = kwargs['pop']
        # else:
        j.population = []


        if j.initial_sample:
            j.pid_cls.initial_sample = j.pid_cls.generate_initial_sample()

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

            self.vis.ft(run_ft)  # Plot run-specific trend

        # Log optimizer average completion time seconds
        self.settings['opt'][kwargs['oid']]['avg_cts'] = total_cts / self.settings['gen']['runs_per_optimizer']

        # Execute problem-specific tasks upon optimization completion, for e.g. generate gantt chart of best schedule
        problem.post_processing(oid=kwargs['oid'])

    def load_components(self):
        pass
        # load components

    def log_optimizer_fitness(self, j):
        if j.rbest.fitness < j.gbest.fitness:
            j.gbest = copy.deepcopy(j.rbest)

        j.gft.append(j.rbest.fitness)

    def summary(self):
        lg.msg(logging.INFO, 'Basic Statistics')
        # need to show all optimzier trends per problem
        gft = {}
        for j in self.jobs:
            gft[j.pid] = {}
            gft[j.pid][j.oid] = j.gft

        for k, v in gft.items():
            self.vis.fitness_trend_all_optimizers(v)
            summary = Stats.get_summary(v)
            # optimizers[k]['avg_cts'], 'lb_diff_pct': optimizers[opt]['lb_diff_pct'], 'ub_diff_pct':
            # optimizers[opt]['ub_diff_pct']}
            # lg.msg(logging.INFO,
            #        'Optimiser\tMin Fitness\tMax Fitness\tAvg Fitness\tStDev\tWilcoxon\tLB Diff %\tUB Diff %\tAvg Cts')
            # for k, v in summary_results.items():
            #     lg.msg(logging.INFO, '{}\t\t{}\t\t{}\t\t{}\t\t{}\t{}\t\t{}\t\t{}\t\t{}'.format(
            #         str(k), str(v['minf']), str(v['maxf']), str(v['mean']), str(v['stdev']), str(v['wts']),
            #         str(v['lb_diff_pct']), str(v['ub_diff_pct']), str(round(v['avg_cts'], 3))))
