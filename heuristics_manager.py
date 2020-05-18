from hopjob import HopJob
import logging
from utilities import logger as lg
from utilities.helper import Helper
from utilities.stats import Stats
from utilities.visualisation import Visualisation
from optimizers.particle import Particle
from optimizers.variator import Variator
from optimizers.crossover import Crossover
from random import Random
import os
import sys
import yaml
from itertools import product
import copy
from importlib import import_module
import time
import statistics

script_name = os.path.basename(sys.argv[0]).split('.')


class HeuristicsManager:
    """
    Controller
    """
    def __init__(self, results_path):
        lg.msg(logging.INFO, 'Initialising Heuristics Manager')
        self.results_path = results_path
        self.random = Random()
        #self.random.seed(42)
        self.vis = Visualisation()
        self.settings = self.get_config()
        self.problems_optimizers = []
        self.problems = []
        self.benchmarks = []
        self.optimizers = []
        self.jobs = self.set_jobs()
        self.exec_start_time = 0

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

    @staticmethod
    def log_optimizer_fitness(j):
        if j.rbest.fitness < j.gbest.fitness:
            j.gbest = copy.deepcopy(j.rbest)

        j.gft.append(j.rbest.fitness)

    def set_jobs(self):
        jobs = []
        self.problems = self.get_problems()
        self.benchmarks = self.get_benchmarks()
        self.optimizers = self.get_optimizers()
        self.problems_optimizers = list(product(self.problems, self.optimizers))
        self.problems_optimizers.sort()

        for (pid, oid) in self.problems_optimizers:
            benchmarks = []
            if 'benchmarks' not in self.settings['prb'][pid]:
                benchmarks.append('na')
            else:
                for bid in self.settings['prb'][pid]['benchmarks']:
                    if not self.settings['prb'][pid]['benchmarks'][bid]['enabled']:
                        continue
                    benchmarks.append(bid)
            for bid in benchmarks:
                jobs.append(self.create_job_spec(pid, oid, bid))

        # Sort jobs by problem, benchmark and optimizer
        jobs = sorted(jobs, key=lambda j: (j.pid, j.bid, j.oid), reverse=False)
        return jobs

    def create_job_spec(self, *args):
        # Create new job specification
        job = HopJob()

        # Persisting directory to support problem specific processing like saving Gantt charts
        job.results_path = self.results_path

        # ----- Assign problem, optimizer, benchmark and type
        job.pid, job.oid, job.bid = args
        job.pid_type = self.settings['prb'][job.pid]['type']
        job.pid_desc = self.settings['prb'][job.pid]['description']

        job.oid_type = self.settings['opt'][job.oid]['type']
        job.oid_desc = self.settings['opt'][job.oid]['description']
        job.oid_optimizer = self.settings['opt'][job.oid]['optimizer']

        # ----- Various co-efficients
        # Declared early here in case referenced by optimizers
        if 'inertia_coeff' in self.settings['prb'][job.pid]:
            job.inertia_coeff = self.settings['prb'][job.pid]['inertia_coeff']

        if 'local_coeff' in self.settings['prb'][job.pid]:
            job.local_coeff = self.settings['prb'][job.pid]['local_coeff']

        if 'global_coeff' in self.settings['prb'][job.pid]:
            job.global_coeff = self.settings['prb'][job.pid]['global_coeff']

        if 'decay' in self.settings['opt'][job.oid]:
            job.decay = self.settings['opt'][job.oid]['decay']

        if 'decay_coeff' in self.settings['opt'][job.oid]:
            job.decay_coeff = self.settings['opt'][job.oid]['decay_coeff']

        # ----- Active components
        # Flags indicating problem and optimizer are active. We still build job spec for optimizer, as it may be
        # disabled "standalone", but actively used as low-level heuristic (llh) in hyper-heuristic
        job.pid_enabled = self.settings['prb'][job.pid]['enabled']
        job.oid_enabled = self.settings['opt'][job.oid]['enabled']

        # ----- Low Level Heuristics
        # Set low-level meta-heuristic components for hype heuristic
        if 'low_level_selection_pool' in self.settings['opt'][job.oid]:
            for llh in self.settings['opt'][job.oid]['low_level_selection_pool']:
                job.low_level_selection_pool.append(llh)

        # ----- Computational Budget
        # Set computing resources like number of runs allocated and computational budget
        job.runs_per_optimizer = self.settings['gen']['runs_per_optimizer']
        job.comp_budget_base = self.settings['gen']['comp_budget_base']

        # Problem class is instantiated here as dimension of problem determines allocated total budget
        prob_module = import_module('problems.' + job.pid.lower())
        job.pid_cls = getattr(prob_module, job.pid)(random=self.random, hopjob=job)

        job.budget = job.pid_cls.n * job.comp_budget_base
        job.budget_total = job.budget

        # ----- Iterations Since Last Improvement
        job.iter_last_imp = [job.budget_total for _ in range(job.runs_per_optimizer)]

        # Set low-level heuristic sampling and computational budget
        if 'llh_sample_runs' in self.settings['opt'][job.oid]:
            job.llh_sample_runs = self.settings['opt'][job.oid]['llh_sample_runs']

        if 'llh_sample_budget_coeff' in self.settings['opt'][job.oid]:
            job.llh_sample_budget = self.settings['opt'][job.oid]['llh_sample_budget_coeff'] * job.budget

        if 'llh_budget_coeff' in self.settings['opt'][job.oid]:
            job.llh_budget = int(self.settings['opt'][job.oid]['llh_sample_budget_coeff'] * job.budget)

        # ----- Binary Encoding
        # Define bit length for optimizers like GA that encode between real and binary
        job.bit_computing = self.settings['gen']['bit_computing']

        # ----- Bounds
        # Problems like Rastrigin bound to [-5.12, 5.12] whilst FSSP combinatorial has upper bound size of n dim
        job.pid_lb = self.settings['prb'][job.pid]['lb']
        if self.settings['prb'][job.pid]['ub'] == 'nmax':
            job.pid_ub = job.pid_cls.n
        else:
            job.pid_ub = self.settings['prb'][job.pid]['ub']

        if 'lb' in self.settings['opt'][job.oid]:
            job.oid_lb = self.settings['opt'][job.oid]['lb']

        if 'ub' in self.settings['opt'][job.oid]:
            job.oid_ub = self.settings['opt'][job.oid]['ub']

        # ----- Sampling
        # Optimizers like SA use an initial sample to determine characteristics like starting temp
        if 'initial_sample' in self.settings['opt'][job.oid]:
            job.initial_sample = self.settings['opt'][job.oid]['initial_sample']

        # ----- Population
        # For optimizers like PSO and GA that work with a defined population size, depends on problem dimension and type
        if job.pid_type == 'combinatorial':
            job.initial_pop_size = job.pid_cls.n * 2
        else:
            job.initial_pop_size = job.pid_cls.n * 3

        if 'number_parents' in self.settings['opt'][job.oid]:
            job.number_parents = self.settings['opt'][job.oid]['number_parents']

        if 'number_children' in self.settings['opt'][job.oid]:
            job.number_children = self.settings['opt'][job.oid]['number_children']

        if 'parent_gene_similarity_threshold' in self.settings['opt'][job.oid]:
            job.parent_gene_similarity_threshold = self.settings['opt'][job.oid]['parent_gene_similarity_threshold']

        # ----- Annealing
        if 'reheat' in self.settings['opt'][job.oid]:
            job.reheat = self.settings['opt'][job.oid]['reheat']

        # ----- Instantiate optimizer class
        opt_module = import_module('optimizers.' + job.oid_optimizer.lower())
        job.oid_cls = getattr(opt_module, job.oid_optimizer)(random=self.random, hopjob=job)

        # ----- Instantiate variator and crossover classes
        job.variator_cls = Variator(random=self.random, hopjob=job)
        job.crossover_cls = Crossover(random=self.random, hopjob=job)

        # ----- Generator (solution), Variator (neighbour from solution) and Crossover (parent) instantiation
        if 'generator_comb' in self.settings['opt'][job.oid]:
            job.generator_comb = getattr(job.pid_cls, 'generator_' + self.settings['opt'][job.oid]['generator_comb'])

        if 'generator_cont' in self.settings['opt'][job.oid]:
            job.generator_cont = getattr(job.pid_cls, 'generator_' + self.settings['opt'][job.oid]['generator_cont'])

        if 'variator' in self.settings['opt'][job.oid]:
            job.variator = getattr(job.variator_cls, 'variator_' + self.settings['opt'][job.oid]['variator'])

        if 'crossover' in self.settings['opt'][job.oid]:
            job.crossover = getattr(job.crossover_cls, 'crossover_' + self.settings['opt'][job.oid]['crossover'])

        return job

    def get_problems(self):
        problems = []
        for pid in self.settings['prb']:
            problems.append(pid)
        return problems

    def get_benchmarks(self):
        benchmarks = []
        for pid in self.problems:
            if 'benchmarks' not in self.settings['prb'][pid]:
                benchmarks.append('na')
            else:
                for bid in self.settings['prb'][pid]['benchmarks']:
                    if not self.settings['prb'][pid]['benchmarks'][bid]['enabled']:
                        continue
                    benchmarks.append(bid)

        return benchmarks

    def get_optimizers(self):
        optimizers = []
        for oid in self.settings['opt']:
            optimizers.append(oid)
        return optimizers

    def execute_jobs(self):
        for j in self.jobs:

            if j.pid_enabled and j.oid_enabled:
                pass
            else:
                continue

            j.start_time = time.time()

            if j.bid != 'n/a':
                lg.msg(logging.INFO, 'Benchmark {}'.format(j.bid))
            lg.msg(logging.INFO, 'Optimizing {} with {} ({})'.format(j.pid_desc, j.oid, j.oid_desc))
            lg.msg(logging.INFO, 'Executing {} sample runs'.format(j.runs_per_optimizer))

            for r in range(j.runs_per_optimizer):
                j.run = r
                self.pre_processing(j)  # Controller pre-processing
                j.pid_cls.pre_processing()  # Problem pre-processing
                j.oid_cls.run(jobs=self.jobs)  # Execute optimizer
                self.post_processing(j)  # Controller post-processing

            j.avg_comp_time_s = j.total_comp_time_s / j.runs_per_optimizer

            # Execute problem-specific tasks upon optimization completion e.g. generate Gantt chart of best schedule
            j.pid_cls.post_processing()

            lg.msg(logging.INFO, 'Completed optimizing {} with {} ({})'.format(j.pid_desc, j.oid, j.oid_desc))

        self.summary()

    def pre_processing(self, j):
        lg.msg(logging.INFO, 'Starting optimizer {} run {}'.format(j.oid, str(j.run)))
        self.exec_start_time = time.time()

        j.rft = []
        j.rbest = Particle()
        j.population = []

    def post_processing(self, j):
        j.end_time = time.time()
        j.total_comp_time_s += time.time() - self.exec_start_time

        if isinstance(j.rbest.candidate[0], float) and j.pid_type == 'combinatorial':
            j.rbest.candidate = j.pid_cls.candidate_spv_continuous_to_discrete(j.rbest.candidate)

        lg.msg(logging.INFO, 'Run {} best fitness is {} with candidate {}'.format(j.run, "{:.10f}".format(
            j.rbest.fitness), j.rbest.candidate))
        lg.msg(logging.INFO, 'Completed benchmark {} optimizer {} run {}'.format(j.bid, j.oid, str(j.run)))

        self.log_optimizer_fitness(j)

        filename = self.results_path + '/' + j.pid + ' ' + j.bid + ' ' + j.oid + ' rbest fitness trend run ' + str(j.run)
        self.vis.fitness_trend(j.rft, filename)  # Plot run-specific trend
        Helper.write_to_csv(j.rft, filename + '.csv', header=False)

        if j.run == j.runs_per_optimizer - 1:
            return

        # Reinstate full computational budget for next job run except last run, as budget used in summary reporting
        j.budget = j.budget_total

    def load_components(self):
        pass

    def summary(self):
        lg.msg(logging.INFO, 'Statistics')
        summary = []
        for p in self.problems:
            if not self.settings['prb'][p]['enabled']:
                continue
            for b in self.benchmarks:
                _new_benchmark = True
                gbest_ft = {}
                bdp = {}  # Bounds diff pct
                other = {}
                for o in self.optimizers:
                    if not self.settings['opt'][o]['enabled']:
                        continue
                    for j in self.jobs:
                        if not (j.pid == p and j.bid == b and j.oid == o):
                            continue
                        if _new_benchmark:
                            lg.msg(logging.INFO, 'Summary for problem {} benchmark {}'.format(p, b))
                            _new_benchmark = False
                        gbest_ft[j.oid] = {}
                        gbest_ft[j.oid] = j.gft
                        bdp[j.oid] = {}
                        other[j.oid] = {}
                        other[j.oid]['avg_comp_time_s'] = j.avg_comp_time_s
                        other[j.oid]['budget'] = j.budget_total
                        other[j.oid]['budget_rem'] = j.budget
                        if j.iter_last_imp:
                            other[j.oid]['avg_iter_last_imp'] = int(statistics.mean(j.iter_last_imp))
                        else:
                            other[j.oid]['avg_iter_last_imp'] = 'n/a'

                        if other[j.oid]['avg_iter_last_imp'] != 'n/a':
                            other[j.oid]['budget_no_imp_pct'] = round(((j.budget_total - other[j.oid]['avg_iter_last_imp']) / j.budget_total) * 100, 2)
                        else:
                            other[j.oid]['budget_no_imp_pct'] = 'n/a'
                        other[j.oid]['imp_count'] = j.imp_count
                        if j.bid != 'na':
                            bdp[j.oid] = [j.pid_cls.ilb, j.pid_lb_diff_pct, j.pid_cls.iub, j.pid_ub_diff_pct]
                        else:
                            bdp[j.oid] = ['na'] * 4

                # Only proceed for compiled stats for valid problem/benchmark/optimizer
                if not gbest_ft:
                    continue

                stats_summary = Stats.get_summary(gbest_ft)

                format_spec = "{:>30}" * 16

                cols = ['Optimizer', 'Min Fitness', 'Max Fitness', 'Avg Fitness', 'StDev', 'Wilcoxon', 'LB', 'LB Diff %',
                        'UB', 'UB Diff %', 'Avg Cts', 'Budget', 'Budget Rem', 'Avg Iter Last Imp', 'Budget No Imp %',
                        'Imp Count']
                summary.append(cols)
                lg.msg(logging.INFO, format_spec.format(*cols))

                for k, v in stats_summary.items():
                    lg.msg(logging.INFO, format_spec.format(str(k),
                                                            str(round(v['minf'], 3)),
                                                            str(round(v['maxf'], 3)),
                                                            str(v['mean']),
                                                            str(v['stdev']),
                                                            str(v['wts']),
                                                            str(bdp[k][0]),
                                                            str(bdp[k][1]),
                                                            str(bdp[k][2]),
                                                            str(bdp[k][3]),
                                                            str(round(other[k]['avg_comp_time_s'], 3)),
                                                            other[k]['budget'],
                                                            other[k]['budget_rem'],
                                                            other[k]['avg_iter_last_imp'],
                                                            other[k]['budget_no_imp_pct'],
                                                            other[k]['imp_count']))
                    summary.append([str(k), str(v['minf']), str(v['maxf']), str(v['mean']), str(v['stdev']), str(v['wts']),
                                   str(bdp[k][0]), str(bdp[k][1]), str(bdp[k][2]), str(bdp[k][3]),
                                    str(round(other[k]['avg_comp_time_s'], 3)), other[k]['budget'], other[k]['budget_rem'],
                                    other[k]['avg_iter_last_imp'], other[k]['budget_no_imp_pct'], other[k]['imp_count']])

                # Summary per problem
                Helper.write_to_csv(summary, self.results_path + '/' + p + ' ' + b + ' problem summary.csv')

                # Fitness trend for all optimizers per problem
                filename = self.results_path + '/' + p + ' ' + b + ' all optimizers gbest fitness trend'
                self.vis.fitness_trend_all_optimizers(gbest_ft, filename)
                Helper.write_to_csv(gbest_ft, filename + '.csv')
