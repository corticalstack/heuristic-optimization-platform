from problems.problem import Problem
import logging
from utilities import logger as lg
from utilities.stats import Stats


class FSSP(Problem):
    """
    Flow Shop Scheduling Problem (FSSP)
    """
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)

        self.jobs = {'quantity': 0, 'list': [], 'total_units': []}
        self.machines = {'quantity': 0, 'loadout_times': [], 'lower_bounds_taillard': [], 'assigned_jobs': []}

        # Load benchmark instance
        self.ilb = 0  # Instance lower bound
        self.iub = 0  # Instance upper bound
        self.load_instance()

        # Set n dimensions
        self.n = self.jobs['quantity']

        #self.jobs_set_total_units()
        #self.machines_set_loadout_times()
        #self.machines_set_lower_bounds_taillard()

    def evaluator(self, candidate, budget=1):
        self.machines['assigned_jobs'] = []
        for i in range(0, self.machines['quantity']):
            self.machines['assigned_jobs'].append([])

        for ji, j in enumerate(candidate):
            start_time = 0
            end_time = 0
            for mi, mt in enumerate(self.jobs['list'][j]):
                if self.machines['assigned_jobs'][mi]:
                    if mi == 0:
                        start_time = self.machines['assigned_jobs'][mi][-1][2]
                    else:
                        curr_job_prev_task_end = self.machines['assigned_jobs'][mi][-1][2]
                        prev_job_task_end = self.machines['assigned_jobs'][mi - 1][-1][2]
                        start_time = max(curr_job_prev_task_end, prev_job_task_end)

                end_time = start_time + mt
                self.machines['assigned_jobs'][mi].append((j, start_time, end_time))
                start_time = end_time

        budget -= 1  # Evaluating has a computational cost so reduce budget
        return self.machines['assigned_jobs'][-1][-1][2], budget

    def pre_processing(self):
        pass

    def post_processing(self):
        self.hj.pid_lb_diff_pct, self.hj.pid_ub_diff_pct = Stats.bounds_compare(self.ilb, self.iub, self.hj.gbest.fitness)

        fitness, _ = self.evaluator(self.hj.gbest.candidate)  # set machine assigned jobs to best permutation
        self.vis.solution_representation_gantt(fitness, self.machines, self.jobs, self.hj.results_path + '/' +
                                               self.hj.pid + ' ' + self.hj.oid + ' gbest Gantt chart')

        lg.msg(logging.INFO, 'Machine times for best fitness {}'.format(fitness))
        self.machines_times(self.hj.gbest.candidate)

        lg.msg(logging.INFO, 'Job times for best fitness of {} with permutation {}'.format(fitness, self.hj.gbest.candidate))
        self.jobs_times(self.hj.gbest.candidate)

    def load_instance(self):
        filename = 'benchmarks/fssp/' + self.hj.bid
        with open(filename, 'r') as f:
            line = f.readlines()
            for i, job_detail in enumerate(line):
                job_detail = job_detail.strip('\n')
                if i == 0:
                    self.jobs['quantity'], self.machines['quantity'] = [int(n) for n in job_detail.split()]
                elif i == 1:
                    self.iub, self.ilb = [int(n) for n in job_detail.split()]
                else:
                    self.jobs_add(job_detail)

    def jobs_add(self, jobs):
        job_times = [int(n) for n in jobs.split()]
        for ji, jt in enumerate(job_times):
            try:
                self.jobs['list'][ji].append(jt)
            except IndexError:
                self.jobs['list'].append([jt])

    def jobs_set_total_units(self):
        self.jobs['total_units'] = [sum(j) for j in self.jobs['list']]
        if logging.DEBUG >= self.logger.level:
            for ji, j in enumerate(self.jobs['total_units']):
                lg.msg(logging.DEBUG, 'Job {} allocated {} time units'.format(ji, j))

    def jobs_times(self, permutation):
        total_idle_time = 0
        _ = self.evaluator(permutation)  # Evaluate best permutation to set machines

        format_spec = "{:>15}" * 4
        lg.msg(logging.INFO, format_spec.format('Job', 'Start Time', 'Finish Time', 'Idle Time'))
        for pi, p in enumerate(permutation):
            start_time = 0
            end_time = 0
            idle_time = 0
            for ji, j in enumerate(self.machines['assigned_jobs']):
                if ji == 0:
                    start_time = j[pi][1]
                    end_time = j[pi][2]
                    continue
                idle_time += j[pi][1] - end_time
                end_time = j[pi][2]
            lg.msg(logging.INFO, format_spec.format(str(p), str(start_time), str(end_time), str(idle_time)))
            total_idle_time += idle_time
        lg.msg(logging.INFO, 'Jobs total idle time is {}'.format(total_idle_time))

    def machines_set_loadout_times(self):
        for m in range(self.machines['quantity']):
            loadout = sum(i[m] for i in self.jobs['list'])
            self.machines['loadout_times'].append(loadout)
            lg.msg(logging.DEBUG, 'Machine {} loaded with {} time units'.format(m, loadout))

    def machines_set_lower_bounds_taillard(self):
        for m in range(self.machines['quantity']):
            lb = self.machines['loadout_times'][m]
            minimum_before_machine_start = []
            minimum_after_machine_start = []
            for j in self.jobs['list']:
                if m > 0:
                    minimum_before_machine_start.append(sum(j[:m]))
                if m < self.machines['quantity']:
                    minimum_after_machine_start.append(sum(j[m+1:]))
            if minimum_before_machine_start:
                lb += min(minimum_before_machine_start)
            if minimum_after_machine_start:
                lb += min(minimum_after_machine_start)
            self.machines['lower_bounds_taillard'].append(lb)
            lg.msg(logging.DEBUG, 'Machine {} Taillard lower bound is {} time units'.format(m, lb))

        lg.msg(logging.INFO, 'Calculated Taillard benchmark instance lower bound (max) is {} time units'.format(
            max(self.machines['lower_bounds_taillard'])))

        if max(self.machines['lower_bounds_taillard']) != self.ilb:
            lg.msg(logging.WARNING, 'Calculated Taillard instance benchmark ({}) != lb in benchmark instance file '
                                        '({})'.format(max(self.machines['lower_bounds_taillard']), self.ilb))

    def machines_times(self, permutation):
        total_idle_time = 0
        _ = self.evaluator(permutation)
        format_spec = "{:>15}" * 4
        lg.msg(logging.INFO, format_spec.format('Machine', 'Start Time', 'Finish Time', 'Idle Time'))

        # Calculate idle time from list tuples as start time(m+1) - finish time(m). Include last machine start time
        for mi, m in enumerate(self.machines['assigned_jobs']):
            finish_time = m[-1][2]
            idle_time = sum([x[1]-x[0] for x in zip([x[2] for x in m], [x[1] for x in m[1:] + [(0, m[-1][2], 0)]])])
            total_idle_time += idle_time
            lg.msg(logging.INFO, format_spec.format(str(mi), str(m[0][1]), str(finish_time), str(idle_time)))
        lg.msg(logging.INFO, 'Machines total idle time is {}'.format(total_idle_time))
