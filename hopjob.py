from optimizers.particle import Particle


class HopJob:
    def __init__(self):
        self.step = 0
        self.type = None
        self.pid = None
        self.pid_cls = None
        self.pid_lb = 0
        self.pid_ub = 0
        self.oid = None
        self.pid_lb_diff_pct = 0
        self.pid_ub_diff_pct = 0
        self.oid_cls = None
        self.oid_lb = 0
        self.oid_ub = 0
        self.bid = None
        self.comp_budget_base = 0
        self.budget = 0
        self.runs_per_optimizer = 0
        self.bit_computing = 0
        self.initial_sample = False
        self.initial_candidate_size = 0
        self.number_parents = 0
        self.number_children = 0
        self.sample_size_factor = 100  # Usually multiples n dimensions to determine sample size
        self.generator = None
        self.variator = None
        self.rbest = Particle()
        self.gbest = Particle()
        self.rft = []
        self.gft = []
        self.population = []
        self.start_time = 0
        self.end_time = 0
        self.total_comp_time_s = 0
        self.avg_comp_time_s = 0
        self.coeff_inertia = 0
        self.coeff_local = 0
        self.coeff_global = 0




