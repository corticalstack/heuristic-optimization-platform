from problems.problem import Problem
import math


class RASTRIGIN(Problem):
    """
    Rastrigin
    """
    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)

        # Set n dimensions
        self.n = 2

    def pre_processing(self):
        pass

    def post_processing(self):
        pass

    def evaluator(self, candidate, budget=1):
        budget -= 1
        return sum([x**2 - 10 * math.cos(2 * math.pi * x) + 10 for x in candidate]), budget
