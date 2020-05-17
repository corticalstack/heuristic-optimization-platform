class Crossover:
    def __init__(self, **kwargs):
        # Persist current configuration and problem
        self.random = kwargs['random']
        self.hj = kwargs['hopjob']

    def crossover_one_point(self, parent0, parent1):
        if self.hj.pid_type == 'combinatorial':
            cp = self.random.randint(1, (len(parent0) - 1))
            child = parent0[:cp]
            # Add missing genes from second parent
            for c in parent1:
                if c not in child:
                    child.append(c)
        else:
            child = []
            for pi, p in enumerate(parent0):
                cp = self.random.randint(1, (len(p) - 1))
                # Continuous without discrete values is simpler, take slice from each parent
                cv = parent0[pi][:cp] + parent1[pi][cp:]
                child.append(cv)
        return child

    def crossover_two_point(self, parent0, parent1):
        if self.hj.pid_type == 'combinatorial':
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
                # Continuous without discrete values is simpler, take slices from parents
                cv = parent0[pi][0:cp[0]] + parent1[pi][cp[0]:cp[1]] + parent0[pi][cp[1]:]
                child.append(cv)
        return child

    def crossover_sbox(self, parent0, parent1):
        if self.hj.pid_type == 'combinatorial':
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
                    try:
                        index = child.index(-1)
                    except ValueError:
                        break
                    child[index] = c
            for ci, c in enumerate(parent1):
                if c not in child:
                    try:
                        index = child.index(-1)
                    except ValueError:
                        break
                    child[index] = c
        else:
            child = []
            for pi0, p0 in enumerate(parent0):
                _child = [-1] * len(p0)
                cp = self.random.randint(1, (len(p0) - 1))
                pi = 0
                while pi < (len(p0) - 1):
                    if parent0[pi0][pi:pi + 2] == parent1[pi0][pi:pi + 2]:
                        _child[pi:pi + 2] = parent0[pi0][pi:pi + 2]
                        pi += 2
                    else:
                        pi += 1
                for ci, c in enumerate(parent0[pi0][0:cp]):
                    try:
                        index = _child.index(-1)
                    except ValueError:
                        break
                    _child[index] = c
                for ci, c in enumerate(parent1[pi0]):
                    try:
                        index = _child.index(-1)
                    except ValueError:
                        break
                    _child[index] = c
                child.append(_child)
        return child
