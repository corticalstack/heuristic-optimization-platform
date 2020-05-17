import copy


class Variator:
    def __init__(self, **kwargs):
        # Persist current configuration and problem
        self.random = kwargs['random']
        self.hj = kwargs['hopjob']

    def variator_n_exchange(self, candidate):
        """
        Exchange job positions and i and j
        """
        def _exchange(_c):
            ops = self.random.sample(range(0, len(_c)), 2)
            _c[ops[0]], _c[ops[1]] = _c[ops[1]], _c[ops[0]]
            return _c

        candidate = copy.deepcopy(candidate)
        if self.hj.pid_type == 'combinatorial':
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
        if self.hj.pid_type == 'combinatorial':
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
        if self.hj.pid_type == 'combinatorial':
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
        if self.hj.pid_type == 'combinatorial':
            candidate = _to_first(candidate)
        else:
            for c in candidate:
                c = _to_first(c)
        return candidate
