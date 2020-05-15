import logging
from utilities import logger as lg
import struct
from optimizers.particle import Particle
import copy


class Optimizer:
    def __init__(self, **kwargs):
        # Persist current configuration and problem
        self.random = kwargs['random']
        self.hj = kwargs['hopjob']

    def run(self, **kwargs):
        self.pre_processing(**kwargs)
        self.optimize()
        self.post_processing()

    def get_generator(self):
        if self.hj.type == 'combinatorial':
            return self.hj.generator_comb
        else:
            return self.hj.generator_cont

    def binary_to_float(self, binary):
        # Transform bit string to float
        float_vals = []
        for b in binary:
            fv = float(int(''.join([str(i) for i in b]), 2))
            # Rescale float within lower and upper bounds of
            fv = fv / (2 ** self.hj.bit_computing - 1) * (self.hj.pid_ub - self.hj.pid_lb) + self.hj.pid_lb
            float_vals.append(fv)
        return float_vals

    def n_insert(self):
        pass

    def variator_n_exchange(self, candidate):
        """
        Exchange job positions and i and j
        """
        def _exchange(_c):
            ops = self.random.sample(range(0, len(_c)), 2)
            _c[ops[0]], _c[ops[1]] = _c[ops[1]], _c[ops[0]]
            return _c

        candidate = copy.deepcopy(candidate)
        if self.hj.type == 'combinatorial':
            candidate = _exchange(candidate)
        else:
            for c in candidate:
                c = _exchange(c)
        return candidate

    def variator_n_exchange_adjacent(self, candidate):
        """
        Exchange job positions and i and i+1
        """
        def _exchange_adjacent(_c):
            i = self.random.randint(0, len(_c) - 1)
            j = i + 1
            if j > len(_c) - 1:
                j = 0
            _c[i], _c[j] = _c[j], _c[i]
            return _c

        candidate = copy.deepcopy(candidate)
        if self.hj.type == 'combinatorial':
            candidate = _exchange_adjacent(candidate)
        else:
            for c in candidate:
                c = _exchange_adjacent(c)
        return candidate

    def variator_n_remove_insert(self, candidate):
        """
        Remove i and insert at j
        """
        def _remove_insert(_c):
            i = self.random.randint(0, len(_c) - 1)
            j = self.random.randint(0, len(_c) - 1)
            while i == j:
                j = self.random.randint(0, len(_c) - 1)
            _c.insert(j, _c.pop(i))
            return _c

        candidate = copy.deepcopy(candidate)
        if self.hj.type == 'combinatorial':
            candidate = _remove_insert(candidate)
        else:
            for c in candidate:
                c = _remove_insert(c)
        return candidate

    def variator_n_to_first(self, candidate):
        """
        Remove i and move to first
        """
        def _to_first(_c):
            i = self.random.randint(0, len(_c) - 1)
            j = 0
            while i == j:
                i = self.random.randint(0, len(_c) - 1)
            _c.insert(j, _c.pop(i))
            return _c

        candidate = copy.deepcopy(candidate)
        if self.hj.type == 'combinatorial':
            candidate = _to_first(candidate)
        else:
            for c in candidate:
                c = _to_first(c)
        return candidate

    def n_shift(self):
        pass

    def n_sbox(self):
        pass

    def crossover_one_point(self, parent0, parent1):
        if self.hj.type == 'combinatorial':
            cp = self.random.randint(1, (len(parent0) - 1))
            child = parent0[:cp]
            for c in parent1:
                if c not in child:
                    child.append(c)
        else:
            child = []
            for pi, p in enumerate(parent0):
                cp = self.random.randint(1, (len(p) - 1))
                cv = parent0[pi][:cp] + parent1[pi][cp:]
                child.append(cv)
        return child

    def crossover_two_point(self, parent0, parent1):
        if self.hj.type == 'combinatorial':
            cp = sorted(self.random.sample(range(1, len(parent0) - 1), 2))

            # Take 2 slices from first parent and concatenate
            child = parent0[0:cp[0]] + parent0[cp[1]:]

            # List child missing genes, taking their order from second parent
            from_parent1 = []
            for c in parent1:
                if c not in child:
                    from_parent1.append(c)

            # Insert required "genes" starting from cp idx
            for gene in reversed(from_parent1):
                child.insert(cp[0], gene)
        else:
            child = []
            for pi, p in enumerate(parent0):
                cp = sorted(self.random.sample(range(1, len(p) - 1), 2))
                cv = parent0[pi][0:cp[0]] + parent1[pi][cp[0]:cp[1]] + parent0[pi][cp[1]:]
                child.append(cv)
        return child

    def crossover_sbox(self, parent0, parent1):
        if self.hj.type == 'combinatorial':
            child = [-1] * len(parent0)
            cp = self.random.randint(1, (len(parent0) - 1))
            pi = 0
            while pi < (len(parent0) - 1):
                if parent0[pi:pi+2] == parent1[pi:pi+2]:
                    child[pi:pi+2] = parent0[pi:pi+2]
                    pi += 2
                else:
                    pi += 1
            for ci, c in enumerate(parent0[0:cp]):
                if c not in child:
                    index = child.index(-1)
                    child[index] = c
            for ci, c in enumerate(parent1):
                if c not in child:
                    index = child.index(-1)
                    child[index] = c
        else:
            ### JP need to refactor crossovers into embedded defs for comb/cont variations
            child = []
            for pi, p in enumerate(parent0):
                cp = sorted(self.random.sample(range(1, len(p) - 1), 2))
                cv = parent0[pi][0:cp[0]] + parent1[pi][cp[0]:cp[1]] + parent0[pi][cp[1]:]
                child.append(cv)
        return child

    def pre_processing(self, **kwargs):
        # self.hj.fitness_trend = []
        #
        # # Set global best single particle if passed
        # #if 'gbest' in kwargs:
        # #    self.gbest = kwargs['gbest']
        # #else:
        # self.hj.gbest = Particle()
        #
        # # Set population of particles if passed
        # #if 'pop' in kwargs:
        # #    self.population = kwargs['pop']
        # #else:
        # self.hj.population = []
        #
        # if self.hj.initial_sample:
        #     self.hj.pid_cls.initial_sample = self.hj.pid_cls.generate_initial_sample()

        pass

    def post_processing(self):
        lg.msg(logging.DEBUG, 'Computational budget remaining is {}'.format(self.hj.budget))

