from optimizers.particle import Particle


class HopJob:
    def __init__(self):
        self.step = 0
        self.pid = None
        self.pid_cls = None
        self.pid_lb = 0
        self.pid_ub = 0
        self.oid = None
        self.oid_cls = None
        self.oid_lb = 0
        self.oid_ub = 0
        self.bid = None
        self.comp_budget_base = 0
        self.budget = 0
        self.runs_per_optimizer = 0
        self.initial_sample = False
        self.sample_size_factor = 100  # Usually multiples n dimensions to determine sample size
        self.generator = None
        self.gbest = Particle()
        self.population = []
        self.start_time = 0
        self.end_time = 0
        self.comp_time_s = 0
        self.fitness_trend = []
        self.gbest_fitness_trend = []




